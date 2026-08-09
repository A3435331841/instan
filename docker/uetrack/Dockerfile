ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="GRT-360 seam-aware UETrack"
LABEL org.opencontainers.image.source="https://github.com/A3435331841/instan"
LABEL grt360.uetrack.commit="fd13b0eaf16d51536008295f3b27807c69eaad50"
LABEL grt360.checkpoint.sha256="1d34778a41c553e3a5e17829d33df4a644f7c948b054a64f46e02fa99558b901"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/uetrack

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libglib2.0-0 libgl1 libjpeg-turbo8 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/uetrack/requirements.txt /opt/grt360/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/grt360/requirements.txt

# These three inputs are deliberately outside Git because they contain the
# pinned upstream source, licensed model weights, and CLIP cache respectively.
COPY tools_local/uetrack_docker/external/UETrack/ /opt/uetrack/
COPY tools_local/uetrack_docker/models/uetrack_base.tar /opt/models/uetrack_base.tar
COPY tools_local/uetrack_docker/clip_cache/ViT-L-14.pt /root/.cache/clip/ViT-L-14.pt
COPY integrations/uetrack/ /opt/grt360/integrations/uetrack/

RUN python /opt/grt360/integrations/uetrack/install_adapter.py \
        --workspace /opt/uetrack \
        --output-root /opt/uetrack_output \
        --checkpoint /opt/models/uetrack_base.tar \
    && python -c "import clip, cv2, timm, torch; print(torch.__version__)"

WORKDIR /opt/uetrack
ENTRYPOINT ["python", "/opt/uetrack/_grt360_file_protocol.py"]
