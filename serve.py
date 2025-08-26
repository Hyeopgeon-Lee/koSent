# -*- coding: utf-8 -*-
"""
===============================================================================
Sentiment API (FastAPI) — 스팸 API와 동일 엔드포인트/스키마
-------------------------------------------------------------------------------
기능 요약
- 단건/배치 예측 엔드포인트(/predict, /predict/batch), 헬스(/health)
- 일관된 JSON 오류 응답, CORS 허용
- 학습 때 사용한 custom_standardize 등록(ㅋ/ㅎ/!/?/... 반복 압축 포함)
- Windows cp949 환경에서도 안전하도록 .keras 내부 vocab 파일을 자동 UTF-8 재패킹
- 임계값: THRESHOLD(env) > eval_meta(th_best/threshold) > 0.5
- 라벨 값: 기본 'pos'/'neg' (env LABEL_POS/LABEL_NEG 제공. 단, 응답 스키마가 pydantic
  Literal['pos','neg']로 고정되어 있으므로 커스텀 라벨을 지정한 경우에도 응답의 label
  필드는 'pos'/'neg'로 반환됩니다. 커스텀 라벨은 /health 응답 등 메타 정보 용도로만 사용)

구동 팁
- 모델(.keras)에는 TextVectorization과 custom_standardize가 포함되어 있어야 함
- TensorFlow 2.17 / Python 3.10 기준, CPU 환경에서도 동작
- 운영 시 uvicorn 멀티 워커 사용 시, 모델 메모리 사용량 고려
===============================================================================
"""

from typing import List, Literal, Optional, Any, Dict
import os
import json
import logging
import zipfile
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 로깅 설정
#  - uvicorn이 별도 로거를 쓰므로 실제 배포에서 로그가 중복될 수 있음
# -----------------------------------------------------------------------------
logger = logging.getLogger("sentiment-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# Keras 2/3 커스텀 직렬화 등록
#  - 학습과 서빙에서 동일한 이름/로직의 표준화 함수가 필요
# -----------------------------------------------------------------------------
try:
    register = keras.saving.register_keras_serializable   # Keras 3 (TF 2.20+)
except AttributeError:
    from tensorflow.keras.utils import register_keras_serializable as register  # Keras 2 (TF 2.17)

@register(package="preproc", name="custom_standardize")
def custom_standardize(x: tf.Tensor) -> tf.Tensor:
    """
    TextVectorization 표준화 함수(학습·평가·서빙 공통):
    - 소문자화
    - URL/EMAIL 토큰화
    - ㅋ/ㅎ/!/?/... 반복 문자 압축(백참조 없이 RE2 호환)
    - 허용 문자만 유지(영/숫/한글+자모, 공백/일부 구두점)
    """
    x = tf.strings.lower(x)
    x = tf.strings.regex_replace(x, r'(https?://\S+|www\.\S+)', ' URL ')
    x = tf.strings.regex_replace(x, r'\S+@\S+\.\S+', ' EMAIL ')
    x = tf.strings.regex_replace(x, r'ㅋ+', 'ㅋ')
    x = tf.strings.regex_replace(x, r'ㅎ+', 'ㅎ')
    x = tf.strings.regex_replace(x, r'!+', '!')
    x = tf.strings.regex_replace(x, r'\?+', '?')
    x = tf.strings.regex_replace(x, r'\.+', '.')
    x = tf.strings.regex_replace(x, r"[^0-9a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ\s\.\,\!\?%:/@_\-]", " ")
    x = tf.strings.regex_replace(x, r"\s+", " ")
    return tf.strings.strip(x)

# -----------------------------------------------------------------------------
# .keras UTF-8 자동 교정 로더
#  - 일부 환경(특히 Windows)에서 자산 텍스트가 cp949로 저장된 경우 로드 실패 방지
# -----------------------------------------------------------------------------
def _repack_keras_zip_utf8(src_path: str) -> str:
    """
    assets/*vocab* 텍스트 파일이 cp949로 기록된 경우 UTF-8로 재패킹하여 임시 .keras 경로 반환.
    """
    tmpdir = tempfile.mkdtemp(prefix="utf8fix_")
    base = os.path.basename(src_path)
    if not base.endswith(".keras"):
        base += ".keras"
    dst = os.path.join(tmpdir, base.replace(".keras", "_utf8fix.keras"))
    with zipfile.ZipFile(src_path, "r") as zin, zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            needs = info.filename.startswith("assets/") and any(
                k in info.filename.lower() for k in ("vocab", "vocabulary", "string_lookup")
            )
            if needs:
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError:
                    data = data.decode("cp949").encode("utf-8")
            zout.writestr(info, data)
    return dst

def _load_model_with_utf8_fallback(path: str) -> keras.Model:
    """
    모델 로드 시 UTF-8 디코딩 문제 발생하면 cp949→UTF-8로 재패킹 후 재시도.
    """
    try:
        return keras.models.load_model(
            path, compile=False,
            custom_objects={"custom_standardize": custom_standardize},
            safe_mode=False,
        )
    except Exception as e:
        msg = (str(e) or "").lower()
        if ("codec can't decode byte" in msg) or ("invalid start byte" in msg) or ("utf-8" in msg and "decode" in msg):
            fixed = _repack_keras_zip_utf8(path)
            return keras.models.load_model(
                fixed, compile=False,
                custom_objects={"custom_standardize": custom_standardize},
                safe_mode=False,
            )
        raise

# -----------------------------------------------------------------------------
# 설정값(환경변수 → 기본값)
# -----------------------------------------------------------------------------
MODEL_PATH      = os.getenv("MODEL_PATH", "model/mySentimentModel.keras")
EVAL_META_PATH  = os.getenv("EVAL_META_PATH", "model/eval_meta.json")
SCORE_DECIMALS  = int(os.getenv("SCORE_DECIMALS", "6"))
TEXT_MAX_CHARS  = int(os.getenv("TEXT_MAX_CHARS", "5000"))
BATCH_MAX_ITEMS = int(os.getenv("BATCH_MAX_ITEMS", "256"))

# 응답 라벨 문구(스키마는 Literal['pos','neg']이므로 실제 반환값은 'pos'/'neg' 고정)
LABEL_POS = os.getenv("LABEL_POS", "pos")
LABEL_NEG = os.getenv("LABEL_NEG", "neg")

def _load_threshold() -> float:
    """
    임계값 결정 우선순위:
    1) 환경변수 THRESHOLD
    2) eval_meta.json의 metrics.val.th_best 또는 최상위 th_best/threshold
    3) 기본값 0.5
    """
    env_th = os.getenv("THRESHOLD")
    if env_th:
        try:
            return float(env_th)
        except ValueError:
            logger.warning("Invalid THRESHOLD env var; falling back to meta/0.5.")
    try:
        with open(EVAL_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # ko_sent_model.py 포맷: meta["metrics"]["val"]["th_best"]
        cur = meta
        for k in ("metrics", "val", "th_best"):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                cur = None
                break
        if isinstance(cur, (int, float)):
            return float(cur)
        # 기타 포맷(e.g., eval_sent.py): 최상위 키들
        for k in ("th_best", "threshold"):
            if k in meta:
                return float(meta[k])
    except FileNotFoundError:
        pass
    return 0.5

THRESHOLD = _load_threshold()

# -----------------------------------------------------------------------------
# 모델 로드
#  - 로드 실패 시에도 앱은 기동(health는 ok=false+에러 노출, predict는 503)
# -----------------------------------------------------------------------------
model = None
_model_load_error: Optional[str] = None
try:
    model = _load_model_with_utf8_fallback(MODEL_PATH)
    logger.info("Model loaded: %s", MODEL_PATH)
except Exception as e:
    _model_load_error = f"{type(e).__name__}: {e}"
    logger.exception("Failed to load model: %s", _model_load_error)

# -----------------------------------------------------------------------------
# Pydantic 스키마 (스팸 API와 동일 필드 구성)
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., description="분류할 문장")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="분류할 문장 리스트")

class PredictResponse(BaseModel):
    # 스키마 레벨에서는 'pos'/'neg'로 고정. 운영상 커스텀 라벨을 써야 하면
    # 스키마를 str로 완화하거나, 사양을 명확히 바꾸는 것을 권장.
    label: Literal["pos", "neg"]
    score: float = Field(..., ge=0.0, le=1.0)
    score_str: Optional[str] = None

class BatchItem(BaseModel):
    index: int
    label: Literal["pos", "neg"]
    score: float
    score_str: Optional[str] = None

# -----------------------------------------------------------------------------
# 공통 유틸
# -----------------------------------------------------------------------------
def error_body(kind: str, message: str, details: Optional[Any] = None) -> Dict[str, Any]:
    """모든 오류를 동일한 JSON 포맷으로 반환하기 위한 헬퍼."""
    body = {"ok": False, "error": {"type": kind, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return body

def fmt_score(p: float) -> str:
    """표시용 소수 자리수 적용."""
    return f"{p:.{SCORE_DECIMALS}f}"

def ensure_ready():
    """모델 로드 실패 상태에서 예측 호출 시 503 응답."""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_model_load_error or 'unknown error'}")

def validate_text(s: str) -> str:
    """단건 입력 검증: 타입, 공백 제거, 빈 문자열, 길이 제한."""
    if s is None or not isinstance(s, str):
        raise HTTPException(status_code=400, detail="Field 'text' must be a string.")
    s2 = s.strip()
    if not s2:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    if len(s2) > TEXT_MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Text too long (>{TEXT_MAX_CHARS} chars).")
    return s2

def validate_texts(lst: List[str]) -> List[str]:
    """배치 입력 검증: 리스트 타입/최대 개수/개별 항목 에러 위치 보고."""
    if not isinstance(lst, list) or len(lst) == 0:
        raise HTTPException(status_code=400, detail="Field 'texts' must be a non-empty list.")
    if len(lst) > BATCH_MAX_ITEMS:
        raise HTTPException(status_code=400, detail=f"Too many items (>{BATCH_MAX_ITEMS}).")
    out = []
    for i, s in enumerate(lst):
        try:
            out.append(validate_text(str(s)))
        except HTTPException as e:
            # 어떤 인덱스에서 실패했는지 명시
            raise HTTPException(status_code=e.status_code, detail=f"[index {i}] {e.detail}")
    return out

def predict_probs(texts: List[str]) -> np.ndarray:
    """
    모델 추론:
    - 모델에 TextVectorization이 포함되어 있으므로 문자열 그대로 입력
    - training=False로 드롭아웃 등 비활성화
    """
    x = tf.constant(texts, dtype=tf.string)
    return model(x, training=False).numpy().ravel()

# -----------------------------------------------------------------------------
# FastAPI 앱 구성
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Korean Sentiment API",
    version="1.0.0",
    description="TensorFlow/Keras 감성(긍/부) 분류기 — 스팸 API와 동일 인터페이스.",
)

# CORS 설정
_allow = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = ["*"] if _allow.strip() == "*" else [o.strip() for o in _allow.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 전역 예외 핸들러: 일관된 오류 포맷 적용
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def req_validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=error_body("RequestValidationError", "Invalid request body.", details=exc.errors()))

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=error_body("HTTPException", exc.detail))

@app.exception_handler(Exception)
async def catch_all_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content=error_body("InternalServerError", f"{type(exc).__name__}: {exc}"))

# -----------------------------------------------------------------------------
# 라우트
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    서버/모델 상태 점검:
    - ok: 모델 로드 여부
    - threshold: 현재 운영 임계값
    - eval_meta 일부 지표 노출(roc_auc/pr_auc/th_best)
    """
    info = {
        "ok": True if model is not None else False,
        "model_path": MODEL_PATH,
        "threshold": THRESHOLD,
        "label_pos": LABEL_POS,
        "label_neg": LABEL_NEG,
        "text_max_chars": TEXT_MAX_CHARS,
        "batch_max_items": BATCH_MAX_ITEMS,
        "cors_allow_origins": allow_origins,
    }
    if model is None:
        info["error"] = _model_load_error
    # eval 메타 파일이 존재한다면 핵심 지표 일부만 노출
    try:
        with open(EVAL_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        info["eval_meta"] = {
            "roc_auc": meta.get("roc_auc") or (((meta.get("metrics") or {}).get("val") or {}).get("roc_auc")),
            "pr_auc":  meta.get("pr_auc")  or (((meta.get("metrics") or {}).get("val") or {}).get("pr_auc")),
            "th_best": meta.get("th_best") or (((meta.get("metrics") or {}).get("val") or {}).get("th_best")),
        }
    except FileNotFoundError:
        pass
    return info

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    단건 예측:
    - 입력 검증 → 확률 계산 → 임계값 기준 라벨 결정
    - 응답 스키마 제약으로 label은 'pos'/'neg'로 고정 반환
    """
    ensure_ready()
    text = validate_text(req.text)
    try:
        prob = float(predict_probs([text])[0])
        # 스키마 Literal 제약으로 응답은 'pos'/'neg' 고정
        label = "pos" if prob >= THRESHOLD else "neg"
        return {"label": label, "score": prob, "score_str": fmt_score(prob)}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {type(e).__name__}: {e}")

@app.post("/predict/batch", response_model=List[BatchItem])
def predict_batch(req: BatchPredictRequest):
    """
    배치 예측:
    - 리스트 입력 검증 → 일괄 추론 → index 보존하여 반환
    - 응답 스키마 제약으로 label은 'pos'/'neg' 고정 반환
    """
    ensure_ready()
    texts = validate_texts(req.texts)
    try:
        probs = predict_probs(texts)
        out: List[BatchItem] = []
        for i, p in enumerate(probs.astype(float).tolist()):
            out.append({"index": i, "label": ("pos" if p >= THRESHOLD else "neg"), "score": p, "score_str": fmt_score(p)})
        return out
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {type(e).__name__}: {e}")

# 편의 엔드포인트(스팸 API와 동일): text 또는 texts 중 하나만 받으면 자동 분기
class PredictFlexible(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

@app.post("/predict-flex")
def predict_flex(req: PredictFlexible):
    ensure_ready()
    if req.text is None and not req.texts:
        raise HTTPException(status_code=422, detail="Provide 'text' or 'texts'.")
    if req.text is not None:
        return predict(PredictRequest(text=req.text))
    else:
        return predict_batch(BatchPredictRequest(texts=req.texts))  # type: ignore

# -----------------------------------------------------------------------------
# 로컬 직접 실행(uvicorn)
#  - 운영에서는: uvicorn serve:app --host 0.0.0.0 --port 8000 --workers N
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
