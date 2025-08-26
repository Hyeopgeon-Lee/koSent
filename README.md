# Korean Sentiment (Ko-Sent) — TF/Keras + FastAPI

한국어(혼합 문장 포함) 텍스트를 **긍정/부정**으로 분류하는 프로젝트입니다.  
TensorFlow/Keras(TextVectorization 포함)로 학습한 모델을 FastAPI로 서빙합니다.

---

## ✨ 주요 특징

- **엔드투엔드 파이프라인**: 학습 → 평가 → API 서빙
- **인코딩 안전**: Windows(cp949) 환경에서도 안전하도록 UTF-8 모드/폴백 처리
- **전처리 일관성**: 학습/평가/서빙 모두 동일한 `custom_standardize` 등록(ㅋ/ㅎ/!/?/… 반복 압축 포함)
- **운영 친화**: CORS, 일관된 오류 응답, 환경변수 기반 임계값(threshold)
- **컨테이너화**: Docker 이미지 제공 (`hyeopgeonlee/ko-sent`)

---

## 📦 포함된 구성요소

- **학습 스크립트**: `ko_sent_model.py`  
  - 데이터 로드(UTF-8 우선), **그룹 기반 8:2 분할**(동일 문장은 같은 split)  
  - TextVectorization **학습 분할만 adapt** (검증 누수 방지)  
  - TextCNN 멀티브랜치(커널 3/4/5), AdamW + ReduceLROnPlateau  
  - 조기종료/체크포인트, 최종 모델 `.keras` 저장  
  - 검증에서 **ROC-AUC/PR-AUC**, **F1 최대 임계값(th_best)** 계산 → `model/eval_meta.json`  
  - 학습/검증 **정확도·손실, ROC, PR** 그래프 `model/*.png` 저장

- **평가 스크립트**: `eval_sent.py`  
  - 저장된 `.keras` 모델을 로드해 **독립 분할**로 재평가  
  - Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC, 혼동행렬  
  - best(=F1 최대) & 0.5 임계값 지표 저장  
  - (윈도우 호환) `.keras` 내부 vocab이 cp949일 때 **자동 UTF-8 재패킹 후 로드**

- **API 서버**: `serve.py` (FastAPI)  
  - 엔드포인트: `/health`, `/predict`, `/predict/batch`, `/predict-flex`  
  - CORS, 일관된 오류 포맷, 임계값/메타 노출  
  - 학습 시 사용한 `custom_standardize`를 **동일 이름/로직**으로 등록  
  - 라벨: 기본 `pos`/`neg` (환경변수로 커스터마이즈 가능)

---

## 🗂 폴더 구조

```
.
├─ data/
│  └─ ratings.txt                 # 'id','document','label' (0/1)
├─ model/
│  ├─ mySentimentModel.keras      # 최종 모델
│  ├─ model_checkpoint.keras      # val 기준 베스트
│  ├─ eval_meta.json              # roc_auc, pr_auc, th_best 등
│  ├─ curve_accuracy.png          # 학습 곡선
│  ├─ curve_loss.png
│  ├─ curve_roc.png
│  └─ curve_pr.png
├─ ko_sent_model.py               # 학습
├─ eval_sent.py                   # 평가
├─ serve.py                       # FastAPI 서버 (스팸 API와 동일 인터페이스)
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## 🧾 데이터셋 형식

- 파일: `data/ratings.txt`  
- 컬럼(헤더 포함):
  - `id`        : 임의 식별자(학습에 미사용)
  - `document`  : 텍스트
  - `label`     : `0`(부정) 또는 `1`(긍정)
- 인코딩: **`utf-8` 권장** (실패 시 `utf-8-sig`→`cp949` 순 폴백)

> 동일한 `document`(정규화 기준)는 **반드시 같은 split**에 묶이도록 그룹 분할합니다. (중복/유사 댓글로 인한 누수 방지)

---

## 🧪 학습(Training)

> **Windows라면** 파이썬을 **UTF-8 모드**로 실행하세요. (Keras가 vocabulary 자산을 UTF-8로 기록)

```bash
# 권장 (Windows/PowerShell)
python -X utf8 ko_sent_model.py

# macOS/Linux
python ko_sent_model.py
```

완료 후 생성물(예시):
- `model/mySentimentModel.keras`
- `model/model_checkpoint.keras`
- `model/eval_meta.json` (예: `{"metrics":{"val":{"roc_auc":0.896,"pr_auc":0.898,"th_best":0.35}}...}`)  
- `model/curve_*.png`

---

## ✅ 평가(Evaluation)

```bash
python -X utf8 eval_sent.py --data data/ratings.txt   --model model/model_checkpoint.keras   --out model/eval_sent.json
```

기능:
- Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC
- best(=F1 최대 임계값) & 0.5 기준 지표, 혼동행렬
- 결과 JSON 저장(`eval_sent.json`)

---

## 🚀 API 서버 실행

### 로컬(Python)

```bash
# Windows 권장: UTF-8 모드
python -X utf8 serve.py

# 또는 uvicorn
uvicorn serve:app --host 0.0.0.0 --port 8000
```

### 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `model/mySentimentModel.keras` | 로드할 Keras 모델 경로 |
| `EVAL_META_PATH` | `model/eval_meta.json` | `th_best` 등 메타 읽기 |
| `THRESHOLD` | (meta의 `th_best` \|\| `0.5`) | 분류 임계값(오버라이드용) |
| `LABEL_POS` / `LABEL_NEG` | `pos` / `neg` | 응답 라벨 문구 |
| `SCORE_DECIMALS` | `6` | 점수 문자열 소수 자리수 |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | 바인딩 주소/포트 |
| `CORS_ALLOW_ORIGINS` | `*` | CORS 허용(콤마 구분) |
| `TEXT_MAX_CHARS` | `5000` | 한 샘플 최대 길이 |
| `BATCH_MAX_ITEMS` | `256` | 배치 최대 건수 |

---

## 📡 API 사용법

FastAPI 문서:
- Swagger UI: `http://<host>:8000/docs`  
- ReDoc: `http://<host>:8000/redoc`

### 공통
- Content-Type: `application/json`
- 라벨 기준: `prob >= THRESHOLD` → `"pos"` else `"neg"`

### 1) Health

`GET /health`

응답 예:
```json
{
  "ok": true,
  "model_path": "model/mySentimentModel.keras",
  "threshold": 0.35,
  "label_pos": "pos",
  "label_neg": "neg",
  "eval_meta": { "roc_auc": 0.896, "pr_auc": 0.898, "th_best": 0.35 }
}
```

### 2) 단건 예측

`POST /predict`

요청:
```json
{ "text": "완전 감동이에요 다시 봐도 좋네요" }
```

응답:
```json
{ "label": "pos", "score": 0.912345, "score_str": "0.912345" }
```

### 3) 배치 예측

`POST /predict/batch`

요청:
```json
{ "texts": ["굿", "재미없고 실망", "감동적이었어요"] }
```

응답:
```json
[
  { "index": 0, "label": "pos", "score": 0.73, "score_str": "0.730000" },
  { "index": 1, "label": "neg", "score": 0.21, "score_str": "0.210000" },
  { "index": 2, "label": "pos", "score": 0.88, "score_str": "0.880000" }
]
```

### 4) Flexible

`POST /predict-flex`  
- `{"text": "..."} | {"texts": [...]}` 둘 중 하나를 보내면 `/predict` 또는 `/predict/batch`로 위임

### 오류 응답 포맷(공통)

```json
{
  "ok": false,
  "error": {
    "type": "RequestValidationError",
    "message": "Invalid request body.",
    "details": []
  }
}
```

---

## 🐳 Docker 이미지

### 이미지 경로
- **Docker Hub:** `docker.io/hyeopgeonlee/ko-sent:latest`  
  (권장: 운영에서는 `:latest` 대신 고정 태그 사용 예: `:v0.1.0`)

### 빠른 시작 (Quick start)

```bash
# 1) 이미지 받기
docker pull hyeopgeonlee/ko-sent:latest

# 2) 실행 (호스트 8000 → 컨테이너 8000)
docker run --rm -p 8000:8000 --name ko-sent-api hyeopgeonlee/ko-sent:latest
```

- API 문서: http://localhost:8000/docs  
- 헬스체크: http://localhost:8000/health

### 모델/메타 파일 외부 마운트 (선택)

이미지에 모델이 포함되어 있지 않거나 바꿔서 쓰고 싶다면, 호스트의 `./model`을 컨테이너 `/app/model`로 마운트하세요.

```bash
# macOS/Linux
docker run -d -p 8000:8000   -v "$(pwd)/model:/app/model"   -e MODEL_PATH=/app/model/mySentimentModel.keras   -e EVAL_META_PATH=/app/model/eval_meta.json   --name ko-sent-api   hyeopgeonlee/ko-sent:latest
```

### Docker Compose 예시

```yaml
version: "3.9"
services:
  ko-sent-api:
    image: hyeopgeonlee/ko-sent:latest
    container_name: ko-sent-api
    ports:
      - "8000:8000"
    environment:
      MODEL_PATH: /app/model/mySentimentModel.keras
      EVAL_META_PATH: /app/model/eval_meta.json
      # THRESHOLD: "0.35"   # 필요 시 오버라이드
      CORS_ALLOW_ORIGINS: "*"
      TF_CPP_MIN_LOG_LEVEL: "2"
      OMP_NUM_THREADS: "1"
      TF_NUM_INTRAOP_THREADS: "1"
      TF_NUM_INTEROP_THREADS: "1"
    volumes:
      - ./model:/app/model:ro
    restart: unless-stopped
```

```bash
docker compose up -d
```

---

## ☸️ Kubernetes 배포 예시

### Deployment & Service

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ko-sent-api
  # namespace: kopo-trainee1
spec:
  replicas: 1
  selector:
    matchLabels: { app: ko-sent-api }
  template:
    metadata:
      labels: { app: ko-sent-api }
    spec:
      containers:
        - name: ko-sent-api
          image: hyeopgeonlee/ko-sent:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: /app/model/mySentimentModel.keras
            - name: EVAL_META_PATH
              value: /app/model/eval_meta.json
            - name: TF_CPP_MIN_LOG_LEVEL
              value: "2"
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 10
            periodSeconds: 20
          resources:
            requests: { cpu: "100m", memory: "256Mi" }
            limits:   { cpu: "500m", memory: "512Mi" }
---
apiVersion: v1
kind: Service
metadata:
  name: ko-sent-api
  # namespace: kopo-trainee1
spec:
  type: ClusterIP
  selector:
    app: ko-sent-api
  ports:
    - port: 8000
      targetPort: 8000
```

### Ingress (예: `https://kosent.k-bigdata.kr`)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ko-sent-api
  namespace: kopo-trainee1
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts: [kosent.k-bigdata.kr]
      secretName: ko-sent-api-tls
  rules:
    - host: kosent.k-bigdata.kr
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ko-sent-api
                port:
                  number: 8000
```

> 배포 후 확인:
> ```bash
> kubectl get ing -n kopo-trainee1 -o wide
> curl -k https://kosent.k-bigdata.kr/health
> ```

---

## 🔖 태그 전략

- 최신: `hyeopgeonlee/ko-sent:latest`  
- 고정 버전 예: `hyeopgeonlee/ko-sent:v0.1.0`  
  → **운영에선 고정 태그를 권장**합니다.

---

## 🛠 트러블슈팅

- **모델 로드 실패**: `MODEL_PATH`/`EVAL_META_PATH` 경로와 `custom_standardize` 이름/로직이 학습과 동일한지 확인  
- **한글/인코딩 문제**: 서버/컨테이너는 기본 UTF-8 + `PYTHONUTF8=1`로 동작  
- **메모리 부족**: 컨테이너 메모리 리미트 상향 또는 스레드/워커 수 조정  
- **임계값 조정**: `eval_meta.json`의 `th_best`를 기본값으로 사용하되, 운영 요구(정밀도/재현율)에 따라 `THRESHOLD` 환경변수로 오버라이드

---

## 🔐 라이선스

이 프로젝트는 **Apache License 2.0**을 따릅니다.

---

## 👤 작성자

- 한국폴리텍대학 서울강서캠퍼스 **빅데이터소프트웨어과**  
- **이협건 교수**  
- ✉️ hglee67@kopo.ac.kr  
- 🔗 빅데이터소프트웨어과 입학 상담 **오픈채팅방**: (https://open.kakao.com/o/gEd0JIad)
