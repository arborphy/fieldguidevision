# Strict EfficientNet-B0 ImageNet frozen linear-probe benchmark

- Model: `torchvision/efficientnet_b0-imagenet1k-v1`
- Protocol: 63 species, 1,641 images, individual-disjoint train/validation/test
- Test Top-1: 21.8009%
- Test Top-5: 47.8673%
- Test macro Top-1: 16.0210%
- Best validation-selected C: 100.0
- Backbone trainable parameters: 0
- `embeddings.npz`: all 1,641 normalized frozen embeddings with image IDs and split names

Test labels were not loaded until test predictions were produced.
