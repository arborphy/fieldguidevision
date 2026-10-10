"""Trainable fusion head for frozen EfficientNet and BioCLIP spatial tokens.

The module deliberately keeps species and organ prediction in separate heads.
Species is the primary task; organ is a lower-weight auxiliary task that shapes
the shared representation without replacing the 63-way species objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FusionMode = Literal[
    "efficientnet_only",
    "bioclip_only",
    "concat",
    "bio_queries_eff",
    "eff_queries_bio",
    "bidirectional",
    "bidirectional_final_only",
]


class FeedForward(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CrossAttentionUpdate(nn.Module):
    """Update target tokens from source tokens with a near-identity start."""

    def __init__(self, width: int, heads: int, dropout: float, layerscale_init: float) -> None:
        super().__init__()
        self.target_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_scale = nn.Parameter(torch.full((width,), layerscale_init))
        self.ffn = FeedForward(width, dropout)
        self.ffn_scale = nn.Parameter(torch.full((width,), layerscale_init))

    def forward(self, target: Tensor, source: Tensor) -> Tensor:
        query = self.target_norm(target)
        key_value = self.source_norm(source)
        update, _ = self.attention(query, key_value, key_value, need_weights=False)
        target = target + self.attention_scale * self.attention_dropout(update)
        return target + self.ffn_scale * self.ffn(target)


class BidirectionalFusionBlock(nn.Module):
    """Run both directions from the same pre-block state to avoid order bias."""

    def __init__(self, width: int, heads: int, dropout: float, layerscale_init: float) -> None:
        super().__init__()
        self.bio_from_eff = CrossAttentionUpdate(width, heads, dropout, layerscale_init)
        self.eff_from_bio = CrossAttentionUpdate(width, heads, dropout, layerscale_init)

    def forward(
        self, bio: Tensor, eff: Tensor, mode: FusionMode
    ) -> tuple[Tensor, Tensor]:
        old_bio, old_eff = bio, eff
        if mode in {"bio_queries_eff", "bidirectional", "bidirectional_final_only"}:
            bio = self.bio_from_eff(old_bio, old_eff)
        if mode in {"eff_queries_bio", "bidirectional", "bidirectional_final_only"}:
            eff = self.eff_from_bio(old_eff, old_bio)
        return bio, eff


class ClassificationHead(nn.Module):
    def __init__(self, width: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DualEncoderFusion(nn.Module):
    """Cross-attention fusion over frozen spatial features.

    EfficientNet receives a 7x7 final map and an aligned 7x7 mid-level map.
    BioCLIP receives an 8x8 grid made by pooling its final 16x16 patch grid.
    A learned global token is prepended to each branch before fusion.
    """

    def __init__(
        self,
        *,
        eff_mid_dim: int,
        eff_final_dim: int,
        bio_dim: int,
        species_classes: int,
        organ_classes: int,
        mode: FusionMode = "bidirectional",
        width: int = 256,
        heads: int = 4,
        blocks: int = 2,
        dropout: float = 0.1,
        layerscale_init: float = 1e-3,
        eff_tokens: int = 49,
        bio_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.width = width
        self.eff_mid_projection = nn.Linear(eff_mid_dim, width)
        self.eff_final_projection = nn.Linear(eff_final_dim, width)
        self.bio_projection = nn.Linear(bio_dim, width)
        self.eff_position = nn.Parameter(torch.zeros(1, eff_tokens + 1, width))
        self.bio_position = nn.Parameter(torch.zeros(1, bio_tokens + 1, width))
        nn.init.trunc_normal_(self.eff_position, std=0.02)
        nn.init.trunc_normal_(self.bio_position, std=0.02)

        self.blocks = nn.ModuleList(
            [BidirectionalFusionBlock(width, heads, dropout, layerscale_init) for _ in range(blocks)]
        )
        self.fused_species_head = ClassificationHead(2 * width, species_classes, dropout)
        self.fused_organ_head = ClassificationHead(2 * width, organ_classes, dropout)
        self.eff_anchor_species_head = ClassificationHead(width, species_classes, dropout)
        self.bio_anchor_species_head = ClassificationHead(width, species_classes, dropout)
        self.species_residual_gate = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, 1),
        )
        nn.init.normal_(self.fused_species_head.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.fused_species_head.net[-1].bias)
        nn.init.zeros_(self.species_residual_gate[-1].weight)
        nn.init.constant_(self.species_residual_gate[-1].bias, -2.5)
        self.eff_species_head = ClassificationHead(width, species_classes, dropout)
        self.bio_species_head = ClassificationHead(width, species_classes, dropout)
        self.eff_organ_head = ClassificationHead(width, organ_classes, dropout)
        self.bio_organ_head = ClassificationHead(width, organ_classes, dropout)
        self._freeze_inactive_paths()

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _freeze_inactive_paths(self) -> None:
        """Keep ablation parameter counts honest and exclude unused optimizers."""
        if self.mode in {"efficientnet_only", "bioclip_only", "concat"}:
            self._freeze(self.blocks)
        elif self.mode == "bio_queries_eff":
            for block in self.blocks:
                self._freeze(block.eff_from_bio)
        elif self.mode == "eff_queries_bio":
            for block in self.blocks:
                self._freeze(block.bio_from_eff)
        if self.mode == "efficientnet_only":
            self._freeze(self.bio_projection)
            self.bio_position.requires_grad_(False)
            self._freeze(self.bio_species_head)
            self._freeze(self.bio_organ_head)
            self._freeze(self.bio_anchor_species_head)
            self._freeze(self.fused_species_head)
            self._freeze(self.species_residual_gate)
        elif self.mode == "bioclip_only":
            self._freeze(self.eff_mid_projection)
            self._freeze(self.eff_final_projection)
            self.eff_position.requires_grad_(False)
            self._freeze(self.eff_species_head)
            self._freeze(self.eff_organ_head)
            self._freeze(self.eff_anchor_species_head)
            self._freeze(self.fused_species_head)
            self._freeze(self.species_residual_gate)
        else:
            self._freeze(self.eff_anchor_species_head)
        if self.mode == "bidirectional_final_only":
            self._freeze(self.eff_mid_projection)

    @staticmethod
    def _pool(tokens: Tensor) -> Tensor:
        return 0.5 * (tokens[:, 0] + tokens[:, 1:].mean(dim=1))

    def forward(
        self, eff_mid: Tensor, eff_final: Tensor, bio_spatial: Tensor, bio_global: Tensor
    ) -> dict[str, Tensor]:
        use_mid = self.mode != "bidirectional_final_only"
        eff_spatial = self.eff_final_projection(eff_final)
        if use_mid:
            eff_spatial = eff_spatial + self.eff_mid_projection(eff_mid)
        eff_global = eff_spatial.mean(dim=1, keepdim=True)
        eff = torch.cat([eff_global, eff_spatial], dim=1) + self.eff_position

        bio_spatial_projected = self.bio_projection(bio_spatial)
        bio_global_projected = self.bio_projection(bio_global).unsqueeze(1)
        bio = torch.cat([bio_global_projected, bio_spatial_projected], dim=1) + self.bio_position

        pre_eff, pre_bio = self._pool(eff), self._pool(bio)
        if self.mode not in {"efficientnet_only", "bioclip_only", "concat"}:
            for block in self.blocks:
                bio, eff = block(bio, eff, self.mode)
        post_eff, post_bio = self._pool(eff), self._pool(bio)

        if self.mode == "efficientnet_only":
            fused = torch.cat([post_eff, torch.zeros_like(post_eff)], dim=-1)
        elif self.mode == "bioclip_only":
            fused = torch.cat([torch.zeros_like(post_bio), post_bio], dim=-1)
        else:
            fused = torch.cat([post_eff, post_bio], dim=-1)

        eff_anchor = self.eff_anchor_species_head(pre_eff)
        bio_anchor = self.bio_anchor_species_head(pre_bio)
        species_delta = self.fused_species_head(fused)
        if self.mode == "efficientnet_only":
            species_anchor = eff_anchor
            species_gate = fused.new_zeros((fused.shape[0], 1))
            species = species_anchor
        elif self.mode == "bioclip_only":
            species_anchor = bio_anchor
            species_gate = fused.new_zeros((fused.shape[0], 1))
            species = species_anchor
        else:
            species_anchor = bio_anchor
            species_gate = torch.sigmoid(self.species_residual_gate(fused))
            species = species_anchor + species_gate * species_delta

        return {
            "species": species,
            "species_anchor": species_anchor,
            "species_delta": species_delta,
            "species_gate": species_gate,
            "organ": self.fused_organ_head(fused),
            "eff_species": self.eff_species_head(pre_eff),
            "bio_species": self.bio_species_head(pre_bio),
            "eff_organ": self.eff_organ_head(pre_eff),
            "bio_organ": self.bio_organ_head(pre_bio),
            "embedding": F.normalize(fused, dim=-1),
            "eff_branch_weight": fused.new_tensor(0.0 if self.mode == "bioclip_only" else 1.0),
            "bio_branch_weight": fused.new_tensor(0.0 if self.mode == "efficientnet_only" else 1.0),
        }


@dataclass(frozen=True)
class LossWeights:
    organ: float = 0.25
    branch_species: float = 0.20
    branch_organ: float = 0.20
    fusion_gate: float = 0.01


class MultiTaskLoss(nn.Module):
    """Species-anchored objective with separately observable components."""

    def __init__(
        self,
        species_class_weights: Tensor | None = None,
        organ_class_weights: Tensor | None = None,
        weights: LossWeights = LossWeights(),
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("species_class_weights", species_class_weights)
        self.register_buffer("organ_class_weights", organ_class_weights)
        self.weights = weights
        self.label_smoothing = label_smoothing

    def forward(
        self, outputs: dict[str, Tensor], species: Tensor, organ: Tensor
    ) -> dict[str, Tensor]:
        species_main = F.cross_entropy(
            outputs["species"], species,
            weight=self.species_class_weights,
            label_smoothing=self.label_smoothing,
        )
        branch_denominator = outputs["eff_branch_weight"] + outputs["bio_branch_weight"]
        species_branch = (
            outputs["eff_branch_weight"]
            * F.cross_entropy(outputs["eff_species"], species, weight=self.species_class_weights)
            + outputs["bio_branch_weight"]
            * F.cross_entropy(outputs["bio_species"], species, weight=self.species_class_weights)
        ) / branch_denominator
        valid_organ = organ >= 0
        if valid_organ.any():
            organ_main = F.cross_entropy(
                outputs["organ"][valid_organ], organ[valid_organ],
                weight=self.organ_class_weights,
                label_smoothing=self.label_smoothing,
            )
            organ_branch = (
                outputs["eff_branch_weight"] * F.cross_entropy(
                    outputs["eff_organ"][valid_organ], organ[valid_organ],
                    weight=self.organ_class_weights,
                )
                + outputs["bio_branch_weight"] * F.cross_entropy(
                    outputs["bio_organ"][valid_organ], organ[valid_organ],
                    weight=self.organ_class_weights,
                )
            ) / branch_denominator
        else:
            organ_main = outputs["organ"].sum() * 0.0
            organ_branch = organ_main

        species_total = species_main + self.weights.branch_species * species_branch
        organ_total = organ_main + self.weights.branch_organ * organ_branch
        fusion_gate = outputs["species_gate"].mean()
        total = (
            species_total
            + self.weights.organ * organ_total
            + self.weights.fusion_gate * fusion_gate
        )
        return {
            "total": total,
            "species_main": species_main,
            "species_branch": species_branch,
            "organ_main": organ_main,
            "organ_branch": organ_branch,
            "species_total": species_total,
            "organ_total": organ_total,
            "fusion_gate": fusion_gate,
        }


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
