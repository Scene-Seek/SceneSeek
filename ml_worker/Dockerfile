FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    UV_HTTP_TIMEOUT=300 \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    MODEL_ROOT=/opt/models \
    MODEL_PATH=/opt/models/yolo26s.pt \
    YOLO_MODEL=/opt/models/yolo26s.pt \
    SIGLIP_MODEL_ID=google/siglip2-base-patch16-224 \
    HF_HOME=/opt/models/huggingface \
    HF_HUB_CACHE=/opt/models/huggingface/hub \
    TRANSFORMERS_CACHE=/opt/models/huggingface \
    XDG_CACHE_HOME=/opt/models/xdg \
    YOLO_CONFIG_DIR=/opt/models/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.12 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

RUN mkdir -p /opt/models /opt/models/huggingface /opt/models/ultralytics /opt/models/xdg

RUN python - <<'PY'
import os
import shutil
from pathlib import Path

from transformers import SiglipModel, SiglipProcessor
from ultralytics import YOLO

model_path = Path(os.environ["MODEL_PATH"])
model_path.parent.mkdir(parents=True, exist_ok=True)

YOLO("yolo26s.pt", verbose=False)
downloaded_yolo = Path("yolo26s.pt")
if downloaded_yolo.exists():
    shutil.copy2(downloaded_yolo, model_path)

siglip_id = os.environ["SIGLIP_MODEL_ID"]
SiglipModel.from_pretrained(siglip_id)
SiglipProcessor.from_pretrained(siglip_id)
PY

COPY . .

CMD ["faststream", "run", "main:app"]
