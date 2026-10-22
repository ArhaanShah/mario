# Inference image: both envs (torch/timm + TensorFlow 2.10/OCTIP) in one Python 3.10 environment, CPU.
# Weights and trained heads are copied in; nothing is downloaded at runtime.
#
# Build from the repo root, after `fetch_weights.py` and training (see RUNBOOK.md step 6):
#   docker build -f docker/Dockerfile -t mario-dino --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
# Run (mount the challenge CSVs/images and an output directory):
#   docker run --rm --network none -v /path/to/data:/app/data -v $PWD/outputs/infer:/app/outputs/infer \
#     mario-dino --config configs/task1.yaml --task 1
FROM python:3.10-slim

ARG GIT_COMMIT=unknown
ENV MARIO_DINO_GIT_COMMIT=${GIT_COMMIT}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SM_FRAMEWORK=tf.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=""

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/ requirements/
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/docker.lock.txt

# Code, configs, the pinned DF41 clone, encoder/OCTIP weights and trained heads (see .dockerignore).
COPY . .

RUN python -c "import mario_dino.infer, mario_dino.octip_cache" \
    && test -f third_party/mario_challenge_code/utils/scoring.py

ENTRYPOINT ["python", "-m", "mario_dino.infer"]
CMD ["--config", "configs/task1.yaml", "--task", "1"]
