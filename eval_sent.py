# -*- coding: utf-8 -*-
r"""
eval_sent.py
- 학습된 감성 모델(.keras)을 로드하여 동일한 방식(그룹 기반 8:2)의 검증셋에서 정량 평가
- Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC / Confusion Matrix
- 세그먼트 평가: URL / EMAIL / 한글 / ㅋ·ㅎ / 부정어 / 길이대(문자수)
- 최적 임계값(F1 최대) 탐색 및 평가 결과(JSON) 저장

호환
- TensorFlow 2.17(Keras 2.x) ~ 2.20+(Keras 3)
- .keras 내부 assets/*vocab* 가 cp949로 저장된 경우 자동 UTF-8 재패킹 후 로드

주의
- 학습 스크립트(ko_sent_model.py)의 custom_standardize와 "이름/로직"이 동일해야 안전
- 평가 분할은 학습과 동일한 시드(SEED=123)로 그룹 기반 8:2 수행 권장
"""

import sys
# 콘솔 출력 인코딩(UTF-8) — 일부 환경에선 reconfigure 미지원
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
import re
import os
import zipfile
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# 학습 스크립트와 동일 시드 사용(그룹 분할 재현 목적)
SEED = 123

# ------------------------------------------------------------
# 0) Keras 2/3 호환: 직렬화 등록 헬퍼
# ------------------------------------------------------------
try:
    _register = keras.saving.register_keras_serializable    # Keras 3 (TF 2.20+)
except AttributeError:
    from tensorflow.keras.utils import register_keras_serializable as _register  # Keras 2.x (TF 2.17)

# ------------------------------------------------------------
# 1) 학습과 "동일"한 표준화 함수 등록 (RE2 호환, 백참조 미사용)
#    - ko_sent_model.py의 custom_standardize와 이름/로직 일치
# ------------------------------------------------------------
@_register(package="preproc", name="custom_standardize")
def custom_standardize(x: tf.Tensor) -> tf.Tensor:
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

# ------------------------------------------------------------
# 2) .keras UTF-8 자동 교정 로더 (assets/*vocab* cp949 → utf-8 재패킹)
# ------------------------------------------------------------
def _repack_keras_zip_utf8(src_path: str) -> str:
    """
    .keras(zip) 내부 assets/*vocab* 류 텍스트가 cp949일 때 UTF-8로 재패킹.
    임시 .keras 경로를 반환.
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

def _load_model_with_utf8_fallback(path: str) -> Tuple[keras.Model, str]:
    """
    모델 로드 시 UTF-8 디코딩 오류가 발생하면 cp949→UTF-8 재패킹 후 재시도.
    (model, effective_path)를 반환.
    """
    try:
        m = keras.models.load_model(
            path,
            compile=False,
            custom_objects={"custom_standardize": custom_standardize},
            safe_mode=False,
        )
        return m, path
    except Exception as e:
        msg = (str(e) or "").lower()
        if ("codec can't decode byte" in msg) or ("invalid start byte" in msg) or ("utf-8" in msg and "decode" in msg):
            fixed = _repack_keras_zip_utf8(path)
            m = keras.models.load_model(
                fixed,
                compile=False,
                custom_objects={"custom_standardize": custom_standardize},
                safe_mode=False,
            )
            return m, fixed
        raise

# ------------------------------------------------------------
# 3) 데이터 유틸 (UTF-8 우선 로드, 그룹 기반 8:2 분할)
# ------------------------------------------------------------
def read_table_flexible(path: str) -> pd.DataFrame:
    """
    CSV/TSV 등 구분자를 pandas에 자동 추정시켜 로드.
    인코딩은 utf-8 → utf-8-sig → cp949 순으로 시도.
    """
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Failed encodings: utf-8/utf-8-sig/cp949")

def py_standardize(s: str) -> str:
    """
    파이썬 전처리(그룹 분할 및 빈 문장 제거에 사용).
    - RE2 호환: 백참조 미사용
    - 반복 문자(ㅋ, ㅎ, !, ?, .) 압축
    - 허용 문자만 유지(영/숫/한글+자모, 공백/일부 구두점)
    """
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'(https?://\S+|www\.\S+)', ' URL ', s)
    s = re.sub(r'\S+@\S+\.\S+', ' EMAIL ', s)
    s = re.sub(r'ㅋ+', 'ㅋ', s)
    s = re.sub(r'ㅎ+', 'ㅎ', s)
    s = re.sub(r'!+', '!', s)
    s = re.sub(r'\?+', '?', s)
    s = re.sub(r'\.+', '.', s)
    s = re.sub(r"[^0-9a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ\s\.\,\!\?%:/@_\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def grouped_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = SEED):
    """
    그룹 기반 분할:
    - 동일한 norm_text는 반드시 같은 split에 들어가도록 보장
    - 검증 비율(val_ratio)에 따라 val 그룹을 먼저 샘플링
    """
    uniq = pd.unique(df["norm_text"].values)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_ratio)
    val_set = set(uniq[:n_val])
    train = df[~df["norm_text"].isin(val_set)].reset_index(drop=True)
    val   = df[ df["norm_text"].isin(val_set)].reset_index(drop=True)
    return train, val

# ------------------------------------------------------------
# 4) 메트릭/평가 유틸
# ------------------------------------------------------------
def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, th: float) -> Dict[str, Any]:
    """
    주어진 임계값(th)에서 정확도/정밀도/재현율/F1과 혼동행렬 요소를 계산.
    """
    y_pred = (y_prob >= th).astype(np.int32)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / max(1, len(y_true))
    return {"th": th, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def confusion(y_true: np.ndarray, y_prob: np.ndarray, th: float) -> List[List[int]]:
    """
    임계값(th)에서의 혼동행렬 [[TN, FP],[FN, TP]]를 반환.
    """
    y_pred = (y_prob >= th).astype(np.int32)
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    return [[tn, fp], [fn, tp]]

def predict_probs_in_batches(model: keras.Model, texts: List[str], batch: int = 4096) -> np.ndarray:
    """
    메모리 절약을 위해 배치 단위로 예측 확률을 계산.
    """
    out = []
    for i in range(0, len(texts), batch):
        bx = tf.constant(texts[i:i + batch], dtype=tf.string)
        out.append(model(bx, training=False).numpy().ravel())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)

@dataclass
class EvalSummary:
    """
    JSON 저장용 평가 요약 데이터 클래스.
    """
    roc_auc: float
    pr_auc: float
    th_best: float
    f1_best: float
    acc_best: float
    prec_best: float
    rec_best: float
    cm_best: Any
    th_050: float
    acc_050: float
    prec_050: float
    rec_050: float
    f1_050: float
    cm_050: Any
    segments: Dict[str, Any]

# ------------------------------------------------------------
# 5) 메인 루틴
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/ratings.txt",
                  help="CSV/TSV 경로 (id,document,label)")
    p.add_argument("--model", type=str, default="model/mySentimentModel.keras",
                  help="학습 모델(.keras) 경로")
    p.add_argument("--seed", type=int, default=SEED,
                  help="검증 분할 시드(학습과 동일 권장)")
    p.add_argument("--val_ratio", type=float, default=0.2,
                  help="검증 비율(기본 0.2; 학습 8:2와 일치 권장)")
    p.add_argument("--out", type=str, default="model/eval_sent.json",
                  help="결과 저장 경로(JSON)")
    p.add_argument("--batch", type=int, default=4096,
                  help="배치 추론 크기")
    args = p.parse_args()

    # 5-1) 데이터 로드/정리
    df = read_table_flexible(args.data)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if not {"document", "label"}.issubset(df.columns):
        raise ValueError("CSV/TSV에 'document','label' 컬럼이 필요합니다. (현재: %s)" % cols)

    df = df.dropna(subset=["label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["document"] = df["document"].astype(str)
    df["norm_text"] = df["document"].apply(py_standardize)
    df = df[df["norm_text"].str.len() > 0].reset_index(drop=True)

    # 5-2) 그룹 기반 8:2 분할(학습과 시드/로직 동일)
    _, val_df = grouped_split(df, val_ratio=args.val_ratio, seed=args.seed)
    X_val = val_df["document"].values.astype(str)
    y_val = val_df["label"].values.astype(np.int32)
    print(f"[INFO] Validation size: {len(X_val)}")

    # 5-3) 모델 로드(UTF-8 fallback 포함)
    model, eff_model_path = _load_model_with_utf8_fallback(args.model)
    print(f"Loaded model: {args.model}\nEffective path: {eff_model_path}")

    # 5-4) 예측 확률(배치 추론)
    probs = predict_probs_in_batches(model, list(X_val), batch=args.batch)

    # 5-5) 임계값 탐색(F1 최대)
    ths = np.linspace(0.05, 0.95, 19)
    f1s = [metrics_at_threshold(y_val, probs, t)["f1"] for t in ths]
    best_idx = int(np.argmax(f1s))
    th_best = float(ths[best_idx])

    m050  = metrics_at_threshold(y_val, probs, 0.5)
    mbest = metrics_at_threshold(y_val, probs, th_best)

    # 5-6) AUCs
    m_roc = keras.metrics.AUC(curve="ROC"); m_roc.update_state(y_val, probs)
    m_pr  = keras.metrics.AUC(curve="PR");  m_pr.update_state(y_val, probs)
    roc_auc = float(m_roc.result().numpy())
    pr_auc  = float(m_pr.result().numpy())

    # 5-7) 혼동행렬
    cm050  = confusion(y_val, probs, 0.5)
    cmbest = confusion(y_val, probs, th_best)

    # 5-8) 세그먼트 평가
    #  - URL/EMAIL/한글/ㅋ·ㅎ/부정어 패턴에 대한 포함 여부
    #  - 길이대는 문자 수(length) 기준으로 3구간
    url_re    = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
    email_re  = re.compile(r"\S+@\S+\.\S+")
    hangul_re = re.compile(r"[\uac00-\ud7a3]")
    laugh_re  = re.compile(r"[ㅋㅎ]")
    neg_re    = re.compile(r"(안|못|않|없|별로|최악|실망)")

    def mask(arr, pat) -> np.ndarray:
        return np.array([1 if pat.search(t) else 0 for t in arr], dtype=np.int32)

    def seg_metrics(name: str, idx: np.ndarray, th: float):
        if len(idx) == 0:
            return name, None
        m = metrics_at_threshold(y_val[idx], probs[idx], th)
        m["count"] = int(len(idx))
        return name, m

    has_url    = mask(X_val, url_re)
    has_email  = mask(X_val, email_re)
    has_hangul = mask(X_val, hangul_re)
    has_laugh  = mask(X_val, laugh_re)
    has_neg    = mask(X_val, neg_re)

    lengths = np.array([len(t) for t in X_val])
    len_bins = {
        "len<=10": np.where(lengths <= 10)[0],
        "11<=len<=30": np.where((lengths > 10) & (lengths <= 30))[0],
        "len>30": np.where(lengths > 30)[0],
    }

    segments: Dict[str, Any] = {}
    # 두 기준(고정 0.5, best) 모두 저장
    for th_name, th in [("0.50", 0.5), (f"{th_best:.2f}", th_best)]:
        seg_block: Dict[str, Any] = {}
        for k, v in {
            "has_url": np.where(has_url == 1)[0],
            "has_email": np.where(has_email == 1)[0],
            "has_hangul": np.where(has_hangul == 1)[0],
            "has_kek": np.where(has_laugh == 1)[0],
            "has_negation": np.where(has_neg == 1)[0],
            **len_bins,
        }.items():
            name, res = seg_metrics(k, v, th)
            if res:
                seg_block[k] = res
        segments[th_name] = seg_block

    # 5-9) 콘솔 출력 요약
    print("\n=== Global metrics ===")
    print(f"ROC-AUC: {roc_auc:.6f}  PR-AUC: {pr_auc:.6f}")
    print(f"[th=0.50] acc={m050['acc']:.6f}  P={m050['prec']:.6f}  R={m050['rec']:.6f}  F1={m050['f1']:.6f}  "
          f"TN={m050['tn']} FP={m050['fp']} FN={m050['fn']} TP={m050['tp']}")
    print(f"[best]   th={th_best:.3f}  acc={mbest['acc']:.6f}  P={mbest['prec']:.6f}  R={mbest['rec']:.6f}  F1={mbest['f1']:.6f}  "
          f"TN={mbest['tn']} FP={mbest['fp']} FN={mbest['fn']} TP={mbest['tp']}")

    # 5-10) JSON 저장
    summary = EvalSummary(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        th_best=th_best,
        f1_best=mbest["f1"],
        acc_best=mbest["acc"],
        prec_best=mbest["prec"],
        rec_best=mbest["rec"],
        cm_best=cmbest,
        th_050=0.5,
        acc_050=m050["acc"],
        prec_050=m050["prec"],
        rec_050=m050["rec"],
        f1_050=m050["f1"],
        cm_050=cm050,
        segments=segments,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    print(f"\nSaved eval to {args.out}")

if __name__ == "__main__":
    main()
