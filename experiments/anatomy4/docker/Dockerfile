# anatomy4 GPU audit image — derived from the existing BioCLIP CUDA image so
# torch/CUDA match the host and no new driver stack is introduced.
#
#   docker build -t anatomy4-audit:0.1 experiments/anatomy4/docker
#
# Runtime (never touches BioCLIP or the DB containers):
#   docker run --rm --device=nvidia.com/gpu=all --network host \
#     -v <repo>:/workspace -v /srv/fast/anatomy4-shenandoah:/out \
#     -v /home/valkyrie/obs-media-vault:/vault:ro \
#     anatomy4-audit:0.1 python -m experiments.anatomy4.cli ...
FROM arq-bioclip:2.5-vith14-cu130

ENV HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
    OPENCV_IO_ENABLE_OPENEXR=0

RUN pip install --no-cache-dir \
      "transformers>=4.56,<5" \
      opencv-python-headless \
      pillow

WORKDIR /workspace

ENTRYPOINT []
CMD ["python", "--version"]
