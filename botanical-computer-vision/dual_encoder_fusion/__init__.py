"""Controlled EfficientNet + BioCLIP feature fusion experiments."""

from .model import DualEncoderFusion, MultiTaskLoss, count_trainable_parameters

__all__ = ["DualEncoderFusion", "MultiTaskLoss", "count_trainable_parameters"]
