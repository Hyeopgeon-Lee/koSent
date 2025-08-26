# =========================================
# Dockerfile — Korean Sentiment API (CPU)
# =========================================
# 베이스: Python 3.10 (슬림)
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

# 런타임 유틸(헬스체크용 curl) + 일부 런타임 라이브러리
# - libgomp1: 일부 과학계산 패키지에서 필요
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 의존성 먼저 복사 → Docker 레이어 캐시 활용
COPY requirements.txt /app/requirements.txt

# TensorFlow 2.17.0 (CPU) + 서빙 의존성 설치
# - GPU 필요 없으면 tensorflow-cpu로 교체 가능
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# 애플리케이션 코드 & 모델 아티팩트 복사
COPY serve.py /app/serve.py
COPY model/ /app/model/

# (옵션) 환경변수 기본값 — 필요 시 docker run에서 override
ENV MODEL_PATH="model/mySentimentModel.keras" \
    EVAL_META_PATH="model/eval_meta.json" \
    HOST="0.0.0.0" \
    PORT="8000" \
    CORS_ALLOW_ORIGINS="*" \
    THRESHOLD="" \
    LABEL_POS="pos" \
    LABEL_NEG="neg" \
    TEXT_MAX_CHARS="5000" \
    BATCH_MAX_ITEMS="256" \
    OMP_NUM_THREADS="1" \
    TF_NUM_INTRAOP_THREADS="1" \
    TF_NUM_INTEROP_THREADS="1"

# 헬스체크: /health 엔드포인트
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT}/health || exit 1

EXPOSE 8000

# Uvicorn 기동
# - 워커 수는 코어/트래픽에 맞춰 조정 (기본 1)
CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
