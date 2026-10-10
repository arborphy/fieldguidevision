from __future__ import annotations

import torch

from dual_encoder_fusion.model import DualEncoderFusion, MultiTaskLoss


def make_batch(batch: int = 3):
    return (
        torch.randn(batch, 49, 12),
        torch.randn(batch, 49, 20),
        torch.randn(batch, 64, 16),
        torch.randn(batch, 16),
    )


def make_model(mode: str):
    return DualEncoderFusion(
        eff_mid_dim=12, eff_final_dim=20, bio_dim=16,
        species_classes=7, organ_classes=4, mode=mode,
        width=32, heads=4, blocks=2, dropout=0.0,
    )


def test_all_modes_have_expected_outputs():
    for mode in (
        "efficientnet_only", "bioclip_only", "concat", "bio_queries_eff",
        "eff_queries_bio", "bidirectional", "bidirectional_final_only",
    ):
        outputs = make_model(mode)(*make_batch())
        assert outputs["species"].shape == (3, 7)
        assert outputs["organ"].shape == (3, 4)
        assert outputs["embedding"].shape == (3, 64)


def test_bidirectional_updates_both_attention_directions():
    model = make_model("bidirectional")
    outputs = model(*make_batch())
    loss = outputs["species"].sum()
    loss.backward()
    block = model.blocks[0]
    assert block.bio_from_eff.attention.in_proj_weight.grad is not None
    assert block.eff_from_bio.attention.in_proj_weight.grad is not None


def test_one_way_only_uses_requested_direction():
    model = make_model("bio_queries_eff")
    outputs = model(*make_batch())
    outputs["species"].sum().backward()
    block = model.blocks[0]
    assert block.bio_from_eff.attention.in_proj_weight.grad is not None
    assert block.eff_from_bio.attention.in_proj_weight.grad is None
    assert not block.eff_from_bio.attention.in_proj_weight.requires_grad


def test_unknown_organ_is_masked_but_species_still_trains():
    model = make_model("bidirectional")
    outputs = model(*make_batch())
    criterion = MultiTaskLoss()
    losses = criterion(outputs, torch.tensor([0, 1, 2]), torch.tensor([-1, -1, -1]))
    assert losses["organ_main"].item() == 0.0
    assert losses["species_main"].item() > 0.0
    losses["total"].backward()
