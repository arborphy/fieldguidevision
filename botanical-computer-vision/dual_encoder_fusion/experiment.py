"""Training and evaluation utilities shared by local tests and Modal runs."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .model import DualEncoderFusion, LossWeights, MultiTaskLoss, count_trainable_parameters


@dataclass(frozen=True)
class TrainConfig:
    width: int = 256
    heads: int = 4
    blocks: int = 2
    dropout: float = 0.10
    layerscale_init: float = 1e-3
    organ_loss_weight: float = 0.25
    branch_species_weight: float = 0.20
    branch_organ_weight: float = 0.20
    fusion_gate_weight: float = 0.01
    label_smoothing: float = 0.05
    batch_size: int = 32
    epochs: int = 80
    patience: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    gradient_clip: float = 1.0


class TokenDataset(Dataset):
    def __init__(self, cache: dict[str, object], indices: np.ndarray) -> None:
        self.cache = cache
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        index = self.indices[item]
        return {
            "index": index,
            "eff_mid": self.cache["eff_mid"][index],
            "eff_final": self.cache["eff_final"][index],
            "bio_spatial": self.cache["bio_spatial"][index],
            "bio_global": self.cache["bio_global"][index],
            "species": self.cache["species_targets"][index],
            "organ": self.cache["organ_targets"][index],
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sqrt_inverse_class_weights(targets: Tensor, classes: int, max_weight: float = 5.0) -> Tensor:
    valid = targets[targets >= 0]
    counts = torch.bincount(valid, minlength=classes).float()
    positive = counts > 0
    weights = torch.ones(classes)
    weights[positive] = torch.sqrt(counts[positive].sum() / (classes * counts[positive]))
    weights = weights.clamp(max=max_weight)
    return weights / weights[positive].mean()


def make_model(cache: dict[str, object], mode: str, config: TrainConfig) -> DualEncoderFusion:
    return DualEncoderFusion(
        eff_mid_dim=cache["eff_mid"].shape[-1],
        eff_final_dim=cache["eff_final"].shape[-1],
        bio_dim=cache["bio_spatial"].shape[-1],
        species_classes=len(cache["species_names"]),
        organ_classes=len(cache["organ_names"]),
        mode=mode,
        width=config.width,
        heads=config.heads,
        blocks=config.blocks,
        dropout=config.dropout,
        layerscale_init=config.layerscale_init,
        eff_tokens=cache["eff_mid"].shape[1],
        bio_tokens=cache["bio_spatial"].shape[1],
    )


def _forward(model: DualEncoderFusion, batch: dict[str, Tensor], device: torch.device):
    return model(
        batch["eff_mid"].to(device=device, dtype=torch.float32, non_blocking=True),
        batch["eff_final"].to(device=device, dtype=torch.float32, non_blocking=True),
        batch["bio_spatial"].to(device=device, dtype=torch.float32, non_blocking=True),
        batch["bio_global"].to(device=device, dtype=torch.float32, non_blocking=True),
    )


def evaluate(
    model: DualEncoderFusion,
    loader: DataLoader,
    device: torch.device,
    species_classes: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    species_truth, species_probabilities, anchor_probabilities = [], [], []
    species_gates, organ_truth, organ_predictions, indices = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            outputs = _forward(model, batch, device)
            species_truth.extend(batch["species"].numpy().tolist())
            species_probabilities.append(outputs["species"].softmax(-1).cpu().numpy())
            anchor_probabilities.append(outputs["species_anchor"].softmax(-1).cpu().numpy())
            species_gates.extend(outputs["species_gate"].squeeze(-1).cpu().numpy().tolist())
            valid = batch["organ"] >= 0
            organ_truth.extend(batch["organ"][valid].numpy().tolist())
            organ_predictions.extend(outputs["organ"][valid].argmax(-1).cpu().numpy().tolist())
            indices.extend(batch["index"].numpy().tolist())
    y_species = np.asarray(species_truth)
    p_species = np.concatenate(species_probabilities)
    p_anchor = np.concatenate(anchor_probabilities)
    predicted_species = p_species.argmax(axis=1)
    predicted_anchor = p_anchor.argmax(axis=1)
    metrics = {
        "species_accuracy": float(accuracy_score(y_species, predicted_species)),
        "species_macro_f1": float(f1_score(y_species, predicted_species, average="macro", zero_division=0)),
        "species_top5": float(top_k_accuracy_score(
            y_species, p_species, k=min(5, species_classes), labels=np.arange(species_classes)
        )),
        "species_anchor_accuracy": float(accuracy_score(y_species, predicted_anchor)),
        "species_anchor_macro_f1": float(f1_score(
            y_species, predicted_anchor, average="macro", zero_division=0
        )),
        "species_changed_from_anchor": float(np.mean(predicted_species != predicted_anchor)),
        "species_fusion_gate_mean": float(np.mean(species_gates)),
        "organ_accuracy": float(accuracy_score(organ_truth, organ_predictions)) if organ_truth else float("nan"),
        "organ_macro_f1": float(f1_score(
            organ_truth, organ_predictions, average="macro", zero_division=0
        )) if organ_truth else float("nan"),
    }
    rows = [
        {
            "index": int(index),
            "species_target": int(target),
            "species_prediction": int(prediction),
            "species_confidence": float(probability[prediction]),
            "species_anchor_prediction": int(anchor_prediction),
            "species_anchor_confidence": float(anchor_probability[anchor_prediction]),
            "species_fusion_gate": float(gate),
            "species_top5": np.argsort(-probability)[:5].astype(int).tolist(),
        }
        for index, target, prediction, probability, anchor_prediction, anchor_probability, gate in zip(
            indices, y_species, predicted_species, p_species,
            predicted_anchor, p_anchor, species_gates
        )
    ]
    return metrics, rows


def run_training(
    cache: dict[str, object], mode: str, seed: int, config: TrainConfig, device: torch.device
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, Tensor]]:
    seed_everything(seed)
    split = np.asarray(cache["splits"])
    train_indices = np.flatnonzero(split == "train")
    validation_indices = np.flatnonzero(split == "validation")
    test_indices = np.flatnonzero(split == "test")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TokenDataset(cache, train_indices), batch_size=config.batch_size, shuffle=True,
        generator=generator, num_workers=2, pin_memory=True, persistent_workers=True,
    )
    validation_loader = DataLoader(
        TokenDataset(cache, validation_indices), batch_size=2 * config.batch_size,
        shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        TokenDataset(cache, test_indices), batch_size=2 * config.batch_size,
        shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True,
    )
    model = make_model(cache, mode, config).to(device)
    species_weights = sqrt_inverse_class_weights(
        cache["species_targets"][train_indices], len(cache["species_names"])
    ).to(device)
    organ_weights = sqrt_inverse_class_weights(
        cache["organ_targets"][train_indices], len(cache["organ_names"])
    ).to(device)
    criterion = MultiTaskLoss(
        species_weights, organ_weights,
        weights=LossWeights(
            organ=config.organ_loss_weight,
            branch_species=config.branch_species_weight,
            branch_organ=config.branch_organ_weight,
            fusion_gate=config.fusion_gate_weight,
        ),
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score, best_epoch, best_state = -1.0, -1, None
    epochs_without_improvement = 0
    history = []

    for epoch in range(config.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = _forward(model, batch, device)
                components = criterion(
                    outputs,
                    batch["species"].to(device, non_blocking=True),
                    batch["organ"].to(device, non_blocking=True),
                )
            scaler.scale(components["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append({key: float(value.detach().cpu()) for key, value in components.items()})
        scheduler.step()
        validation_metrics, _ = evaluate(
            model, validation_loader, device, len(cache["species_names"])
        )
        epoch_row = {
            "epoch": epoch + 1,
            **{f"train_{key}": float(np.mean([row[key] for row in losses])) for key in losses[0]},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_row)
        score = validation_metrics["species_macro_f1"]
        if score > best_score + 1e-5:
            best_score, best_epoch = score, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_metrics, predictions = evaluate(model, test_loader, device, len(cache["species_names"]))
    result = {
        "mode": mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_species_macro_f1": best_score,
        "trainable_parameters": count_trainable_parameters(model),
        **test_metrics,
        "config": asdict(config),
        "history": history,
    }
    return result, predictions, {key: value.cpu() for key, value in best_state.items()}
