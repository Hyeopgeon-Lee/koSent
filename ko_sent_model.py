# -*- coding: utf-8 -*-
"""
ko_sent_model.py — 안정화 튜닝 + 그래프 (ReduceLROnPlateau만 사용, CosineDecay 제거)

개요
- 한국어 감성(긍정/부정) 이진 분류 모델 학습 스크립트
- 텍스트 전처리: RE2 호환 정규식 기반(백참조 없음), URL/이메일/반복문자 처리
- 데이터 분할: 동일 문장(norm_text 기준)은 반드시 같은 그룹으로 들어가도록 "그룹 기반 8:2 분할"
- 피처화: Keras TextVectorization + 커스텀 표준화 함수(서빙/평가와 동일해야 함)
- 모델 구조: TextCNN(Conv1D 커널 3/4/5 병렬) + GlobalMaxPool + Dense
- 최적화: AdamW(초기 학습률 5e-4, 스케줄 미사용), ReduceLROnPlateau로 학습률 자동 감소
- 콜백: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
- 산출물: model/*.keras, model/train_history.csv, model/curve_*.png, model/eval_meta.json

권장 환경/실행 팁(Windows 포함)
- 파이썬 프로세스를 UTF-8 모드로 실행(한글 로그/자산 인코딩 문제 예방)
  * PowerShell:  python -X utf8 ko_sent_model.py
  * 또는 환경변수: setx PYTHONUTF8 1  (새 셸부터 적용)
- TensorFlow 2.17.0, Python 3.10 기준
- Matplotlib는 "Agg" 백엔드로 설정해 X 서버가 없는 환경에서도 이미지 저장 가능

주의
- 학습/평가/서빙(serve.py)의 custom_standardize는 "동일한 이름/로직"을 유지해야 안전
- .keras 저장 포맷 내부에 텍스트 자산(어휘 등)이 UTF-8로 기록되므로 UTF-8 모드 실행을 권장
"""

import sys

# 표준 출력 스트림 인코딩을 UTF-8로 재설정
# - 일부 환경에서는 reconfigure 미지원일 수 있으므로 예외 무시
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
import json
import pathlib
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# 시각화(그래프 저장용). X 디스플레이가 없는 서버에서도 저장 가능한 백엔드 사용
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0) 난수 시드 고정 (완전 재현은 아님)
#    - CUDA, 병렬 스케줄링 등 외부 요인으로 완벽한 재현은 어려움
# ------------------------------------------------------------
SEED = 123
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# 1) 경로 설정
# ------------------------------------------------------------
DATA_PATH = pathlib.Path("data/ratings.txt")  # 입력 파일: id, document, label
SAVE_DIR  = pathlib.Path("model")             # 산출물 저장 폴더
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2) Keras 2/3 직렬화 호환 (custom_standardize 등록용)
#    - Keras 3: keras.saving.register_keras_serializable
#    - Keras 2: tensorflow.keras.utils.register_keras_serializable
# ------------------------------------------------------------
try:
    register = keras.saving.register_keras_serializable
except AttributeError:
    from tensorflow.keras.utils import register_keras_serializable as register

# ------------------------------------------------------------
# 3) 데이터 로더 (UTF-8 우선 → utf-8-sig → cp949 순서)
#    - pandas가 구분자 자동 추정(sep=None, engine="python")
# ------------------------------------------------------------
def read_table_flexible(path: pathlib.Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("auto", b"", 0, 1, "Failed encodings: utf-8/utf-8-sig/cp949")

# ------------------------------------------------------------
# 4) 파이썬 전처리(그룹분할/빈문장 제거에 사용)
#    - RE2 호환: 백참조(\1 등) 사용하지 않음
#    - 반복문자(ㅋ, ㅎ, !, ?, .) 압축
#    - 허용 문자만 유지(영/숫/한글 + 공백/일부 구두점)
# ------------------------------------------------------------
def py_standardize(s: str) -> str:
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

# ------------------------------------------------------------
# 5) Keras TextVectorization 표준화 함수
#    - 학습/평가/서빙에서 동일한 이름/로직으로 등록되어야 직렬화/역직렬화 안전
#    - 텐서플로우의 정규식 엔진 RE2 기준(백참조 사용 금지)
# ------------------------------------------------------------
@register(package="preproc", name="custom_standardize")
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
# 6) 데이터 준비: 읽기 → 컬럼 검증 → 정규화 텍스트 생성 → 그룹 분할(8:2)
#    - 동일한 norm_text는 반드시 동일 split(훈련/검증)으로 가도록 보장
#    - 텍스트가 비거나 공백만 남은 경우 제거
# ------------------------------------------------------------
print(f"[INFO] Loading: {DATA_PATH}")
df = read_table_flexible(DATA_PATH)

# 열 이름 정규화
df.columns = [c.strip().lower() for c in df.columns]
if not {"document", "label"}.issubset(df.columns):
    raise ValueError(f"필수 컬럼(document, label) 없음: {df.columns.tolist()}")

# 라벨 정리(0/1만 사용)
df = df.dropna(subset=["label"]).copy()
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])].copy()

# 문자열 변환 및 정규화 텍스트 생성
df["document"] = df["document"].astype(str)
df["norm_text"] = df["document"].apply(py_standardize)

# 비어 있는 문장 제거
before = len(df)
df = df[df["norm_text"].str.len() > 0].copy()
print(f"[INFO] Removed empty rows: {before - len(df)}")

# 그룹 기반 분할(동일 norm_text는 같은 그룹)
uniq = pd.unique(df["norm_text"].values)
rng = np.random.RandomState(SEED)
rng.shuffle(uniq)
n_val = int(len(uniq) * 0.2)
val_groups = set(uniq[:n_val])
train_groups = set(uniq[n_val:])

train_df = df[df["norm_text"].isin(train_groups)].reset_index(drop=True)
val_df   = df[df["norm_text"].isin(val_groups)].reset_index(drop=True)
print(f"[INFO] Split -> train: {len(train_df)}, val: {len(val_df)}")
print(f"[INFO] Train pos ratio: {train_df['label'].mean():.4f} | Val pos ratio: {val_df['label'].mean():.4f}")

# 넘파이 배열로 분리
x_train = train_df["document"].values.astype(str)
y_train = train_df["label"].values.astype(np.int32)
x_val   = val_df["document"].values.astype(str)
y_val   = val_df["label"].values.astype(np.int32)

# ------------------------------------------------------------
# 7) tf.data 파이프라인 구성
#    - cache + prefetch로 I/O 병목 완화
# ------------------------------------------------------------
BATCH = 128
AUTO = tf.data.AUTOTUNE

def make_ds(xs, ys, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((xs, ys))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(xs), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH).cache().prefetch(AUTO)

train_ds = make_ds(x_train, y_train, True)
val_ds   = make_ds(x_val,   y_val,   False)

# ------------------------------------------------------------
# 8) TextVectorization
#    - standardize=custom_standardize (학습/서빙 일관성)
#    - adapt는 반드시 "학습 텍스트"로만 수행(검증 누수 방지)
# ------------------------------------------------------------
MAX_TOKENS = 40000     # 어휘 사전 상한(빈도순 상위만 유지)
MAX_LEN    = 160       # 문장 길이(토큰) 고정
vectorize = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize=custom_standardize,
    split="whitespace",
)
print("[INFO] Adapting TextVectorization...")
vectorize.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(512))

# ------------------------------------------------------------
# 9) 모델: TextCNN
#    - 임베딩 → Conv1D(커널 3/4/5) 병렬 → GlobalMaxPool → Concat → Dense
#    - 규제(L2)와 Dropout으로 과적합 완화
# ------------------------------------------------------------
EMBED_DIM = 96
FILTERS   = 96
L2 = keras.regularizers.l2(5e-5)

inp = layers.Input(shape=(1,), dtype=tf.string, name="text")
x = vectorize(inp)                                # [B, L]
x = layers.Embedding(MAX_TOKENS, EMBED_DIM)(x)    # [B, L, D]
x = layers.SpatialDropout1D(0.3)(x)               # 임베딩 수준의 드롭아웃

branches = []
for k in (3, 4, 5):
    b = layers.Conv1D(FILTERS, k, padding="same",
                      activation="relu", kernel_regularizer=L2)(x)
    b = layers.GlobalMaxPooling1D()(b)
    branches.append(b)

x = layers.Concatenate()(branches)                # [B, 3*FILTERS]
x = layers.Dropout(0.6)(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=L2)(x)
out = layers.Dense(1, activation="sigmoid")(x)    # 이진 확률

model = models.Model(inp, out)

# ------------------------------------------------------------
# 10) 옵티마이저/컴파일
#     - AdamW(고정 float lr) + ReduceLROnPlateau로 lr 조절
#     - 메트릭: 정확도 + ROC AUC + PR AUC
# ------------------------------------------------------------
EPOCHS = 8
initial_lr = 5e-4  # settable float lr (Plateau가 필요 시 절반으로 줄임)
optimizer = keras.optimizers.AdamW(
    learning_rate=initial_lr,
    weight_decay=2e-4,
    clipnorm=1.0,
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc", curve="ROC"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
    ],
)

# ------------------------------------------------------------
# 11) 콜백 구성
#     - EarlyStopping: val_pr_auc 기준 정체 시 조기 종료(최고 가중치 복원)
#     - ReduceLROnPlateau: val_pr_auc가 개선되지 않을 때 lr *= 0.5 (최소 1e-5)
#     - ModelCheckpoint: val_pr_auc 기준 베스트 스냅샷 저장(.keras)
#     - CSVLogger: 각 epoch별 로그를 CSV로 저장
# ------------------------------------------------------------
ckpt_path = SAVE_DIR / "model_checkpoint.keras"
csvlog_path = SAVE_DIR / "train_history.csv"

early = keras.callbacks.EarlyStopping(
    monitor="val_pr_auc", mode="max", patience=1, restore_best_weights=True
)
plateau = keras.callbacks.ReduceLROnPlateau(
    monitor="val_pr_auc", mode="max", factor=0.5, patience=1, min_lr=1e-5, verbose=1
)
ckpt = keras.callbacks.ModelCheckpoint(
    filepath=str(ckpt_path),
    monitor="val_pr_auc", mode="max",
    save_best_only=True, save_weights_only=False
)
csvlog = keras.callbacks.CSVLogger(str(csvlog_path), append=False)

# ------------------------------------------------------------
# 12) 학습
# ------------------------------------------------------------
print("[INFO] Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early, plateau, ckpt, csvlog],
)

# ------------------------------------------------------------
# 13) 최종 모델 저장
#     - 전처리(TextVectorization) 포함 전체 모델을 .keras로 저장
# ------------------------------------------------------------
final_path = SAVE_DIR / "mySentimentModel.keras"
model.save(str(final_path))
print("Training finished.")
print(f"Saved model: {final_path}")
print(f"Best checkpoint: {ckpt_path}")

# ------------------------------------------------------------
# 14) 검증셋 평가 및 임계값(th_best) 산출
#     - ROC-AUC / PR-AUC 계산
#     - F1 최대가 되는 임계값을 0.05~0.95 구간에서 탐색
# ------------------------------------------------------------
def collect_probs(ds):
    probs, y_true = [], []
    for xb, yb in ds:
        p = model(xb).numpy().ravel()
        probs.append(p)
        y_true.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(y_true)

probs, yv = collect_probs(val_ds)

m_roc = keras.metrics.AUC(curve="ROC")
m_pr  = keras.metrics.AUC(curve="PR")
m_roc.update_state(yv, probs)
m_pr.update_state(yv, probs)
roc_auc = float(m_roc.result().numpy())
pr_auc  = float(m_pr.result().numpy())

def metrics_at(y_true, y_prob, th: float):
    y_pred = (y_prob >= th).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / len(y_true)
    return f1, prec, rec, acc, tp, fp, tn, fn

ths = np.linspace(0.05, 0.95, 19)
f1_list = [metrics_at(yv, probs, float(t))[0] for t in ths]
th_best = float(ths[int(np.argmax(f1_list))])

# ------------------------------------------------------------
# 15) 곡선 계산/그리기(ROC, PR) 및 학습곡선(Accuracy/Loss)
#     - 저장 경로: model/curve_*.png
# ------------------------------------------------------------
def roc_curve_np(y_true, y_score):
    # 내림차순 정렬 후 누적 계산으로 FPR, TPR 계산
    order = np.argsort(-y_score)
    y_true = y_true[order]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    tpr = tps / (P + 1e-9)
    fpr = fps / (N + 1e-9)
    return fpr, tpr

def pr_curve_np(y_true, y_score):
    # 내림차순 정렬 후 누적 TP/FP로 Precision-Recall 계산
    order = np.argsort(-y_score)
    y_true_s = y_true[order]
    tp = np.cumsum(y_true_s == 1)
    fp = np.cumsum(y_true_s == 0)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (np.sum(y_true_s == 1) + 1e-9)
    return recall, precision

def auc_trapz(x, y):
    return float(np.trapz(y, x))

# Accuracy 곡선
acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
if acc and val_acc:
    plt.figure()
    plt.plot(range(1, len(acc) + 1), acc, label="train")
    plt.plot(range(1, len(val_acc) + 1), val_acc, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    acc_png = SAVE_DIR / "curve_accuracy.png"
    plt.savefig(acc_png, dpi=150, bbox_inches="tight")
    plt.close()
else:
    acc_png = None

# Loss 곡선
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])
if loss and val_loss:
    plt.figure()
    plt.plot(range(1, len(loss) + 1), loss, label="train")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    loss_png = SAVE_DIR / "curve_loss.png"
    plt.savefig(loss_png, dpi=150, bbox_inches="tight")
    plt.close()
else:
    loss_png = None

# ROC 곡선
fpr, tpr = roc_curve_np(yv, probs)
roc_area = auc_trapz(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC={roc_area:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend()
roc_png = SAVE_DIR / "curve_roc.png"
plt.savefig(roc_png, dpi=150, bbox_inches="tight")
plt.close()

# PR 곡선
rec, prec = pr_curve_np(yv, probs)
pr_area = auc_trapz(rec, prec)
f1_vals = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = int(np.nanargmax(f1_vals))
probs_sorted = np.sort(probs)[::-1]
th_est = float(probs_sorted[best_idx])  # PR 상 최대 F1 지점 근사 임계값(참고용)

plt.figure()
plt.plot(rec, prec, label=f"PR AUC={pr_area:.4f}")
plt.scatter([rec[best_idx]], [prec[best_idx]],
            marker="o", label=f"Best F1={float(f1_vals[best_idx]):.4f} @th≈{th_est:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Validation)")
plt.legend()
pr_png = SAVE_DIR / "curve_pr.png"
plt.savefig(pr_png, dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# 16) 평가 메타 저장(JSON)
#     - 서빙(serve.py)에서 임계값 기본값으로 사용 가능
# ------------------------------------------------------------
meta = {
    "task": "korean_sentiment_binary",
    "data": {"train": int(len(train_df)), "val": int(len(val_df))},
    "vectorizer": {"max_tokens": int(MAX_TOKENS), "sequence_length": int(MAX_LEN)},
    "metrics": {"val": {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "th_best": float(th_best)}},
    "artifacts": {
        "final_model": str(final_path),
        "best_checkpoint": str(ckpt_path),
        "history_csv": str(SAVE_DIR / "train_history.csv"),
    },
    "plots": {
        "accuracy": (str(acc_png) if acc_png else None),
        "loss": (str(loss_png) if loss_png else None),
        "roc": str(roc_png),
        "pr": str(pr_png),
    },
    "note": "AdamW(lr=5e-4) + ReduceLROnPlateau; CosineDecay 미사용. 그룹 기반 8:2 분할.",
}
with open(SAVE_DIR / "eval_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Saved eval meta:", SAVE_DIR / "eval_meta.json")
print("Saved plots:")
print(" -", roc_png)
print(" -", pr_png)
if acc_png:
    print(" -", acc_png)
if loss_png:
    print(" -", loss_png)
print(" -", SAVE_DIR / "train_history.csv")
