# anatomy4 detector trainer image.
#
# Derived from the existing anatomy4 audit image (torch/CUDA already match the
# .245 host); only adds Ultralytics + ONNX Runtime. Nothing here touches the
# BioCLIP, OSM sidecar, observations sidecar, or app containers.
#
# Build on .245:
#   docker build -t anatomy4-detector:0.1 experiments/anatomy4_detector/docker
#
# Run (GPU via CDI, host networking for the repo mount; no service restarts):
#   docker run --rm --device=nvidia.com/gpu=all --network host \
#     --user "$(id -u):$(id -g)" \
#     -v <repo>:/workspace -v /srv/fast/anatomy4-runs:/runs \
#     anatomy4-detector:0.1 python -m experiments.anatomy4_detector.cli ...
FROM anatomy4-audit:0.1

ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=0 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN pip install --no-cache-dir \
      "ultralytics==8.4.190" \
      "onnxruntime>=1.18" \
      "onnx>=1.16" \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir "opencv-python-headless>=4.10"

WORKDIR /workspace

ENTRYPOINT []
CMD ["python", "-c", "import torch, ultralytics, onnxruntime; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__, onnxruntime.__version__)"]
