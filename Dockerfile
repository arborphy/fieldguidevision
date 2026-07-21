# Fieldguidevision DeVo gate — Cloud Run image (CPU).
#
# Build context: the fieldguidevision/ repo root.
#   docker build -t fieldguidevision-gate .
#
# Model weights are baked into the image at build time so Cloud Run cold
# starts never hit Hugging Face. The gate serves scripts/serve_growth_form_gate.py
# on $PORT (default 8080) with --device cpu.

FROM python:3.12-slim AS weights

ARG DETECTOR_MODEL=IDEA-Research/grounding-dino-tiny
ARG SEGMENTER_MODEL=facebook/sam2.1-hiera-tiny
ENV HF_HOME=/models

RUN pip install --no-cache-dir huggingface_hub \
 && python -c "from huggingface_hub import snapshot_download; [print('downloaded', r, '->', snapshot_download(r)) for r in ('${DETECTOR_MODEL}', '${SEGMENTER_MODEL}')]"

FROM python:3.12-slim

ARG DETECTOR_MODEL=IDEA-Research/grounding-dino-tiny
ARG SEGMENTER_MODEL=facebook/sam2.1-hiera-tiny

ENV HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    DETECTOR_MODEL=${DETECTOR_MODEL} \
    SEGMENTER_MODEL=${SEGMENTER_MODEL}

WORKDIR /app

# CPU-only torch first so transformers pulls no CUDA wheels (keeps the image
# small enough for Artifact Registry + fast cold starts). Deps mirror the
# `segmentation` extra in pyproject.toml, minus torch (installed above) and
# plus nothing the gate doesn't import.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers opencv-python-headless pillow

# The gate script inserts the repo root on sys.path itself; no pip install
# of the project is required.
COPY segmentation ./segmentation
COPY scripts/serve_growth_form_gate.py ./scripts/serve_growth_form_gate.py

COPY --from=weights /models /models

EXPOSE 8080

CMD exec python scripts/serve_growth_form_gate.py \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --device cpu \
    --detector-model ${DETECTOR_MODEL} \
    --segmenter-model ${SEGMENTER_MODEL}
