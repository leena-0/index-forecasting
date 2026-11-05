# 모델 사용법 (`src/models`)

입력은 공통 특성 패널(`src.pipeline.features.load_panel`, `docs/PIPELINE.md`), 출력은 CLAUDE.md 6.2 형식의 예측 테이블이다.
평가는 공통 평가 코드(`src.evaluate`, `docs/EVALUATION.md`)로만 한다.

## 1. 실행

```bash
# prepare 결과 폴더(features.parquet 포함)가 있어야 한다
.venv/Scripts/python -m src.models --model ewma --h 5 --data-dir data/dev    # → data/preds/EWMA_h5.parquet
.venv/Scripts/python -m src.models --model har  --h 22 --data-dir data/dev   # → data/preds/HAR_h22.parquet
.venv/Scripts/python -m src.models --model ewma --h 5 --lam 0.97                       # → EWMA_0.97_h5 (검증 구간 비교용)
```

모든 분할(train·val·test)의 예측을 만든다. 테스트 구간 평가는 `src.evaluate --split test --final`로 한 번만 한다.
`{model}_h{h}.meta.json`에 설정·계수·git commit과 prepare의 `reportable`이 남는다.

## 2. 모델 (1·2·3단계, CLAUDE.md 6.1) [제안]

| 모델 | 정의 | 설정 |
|---|---|---|
| EWMA | σ²_t = λ·σ²_{t−1} + (1−λ)·rv_d_t, 다음 h일 예측 = σ²_t | λ = 0.94 고정(RiskMetrics). 초기값 = 처음 22개 `rv_d` 평균, 그 전엔 예측 없음. `rv_d` 결측일은 갱신하지 않음 |
| HAR (Corsi, 2009) | log_var = β0 + β1·ln rv_d + β2·ln rv_w + β3·ln rv_m | 전 종목 통합 OLS, 학습 구간(`train`, `embargo == False`)에서 한 번 추정 |
| LightGBM | t일 특성(분산 특성은 로그) → log_var, L2 손실 | 하이퍼파라미터 고정(`lgbm.PARAMS`: 학습률 0.05, 트리 500개, 잎 31, 잎 최소 100행, 행·특성 표본 80%), 튜닝 없음. 결측은 LightGBM이 따로 분기 |
| LSTM | 최근 22거래일 `[ln rv_d, ret, log_turnover]` 시퀀스 → log_var, MSE | 은닉 32, 1층, 에포크 5(고정, 조기 종료 없음), 배치 1024, Adam 1e-3. 학습 구간 통계로 표준화. 창 안에 결측이 있으면 예측 없음 |
| GRU (2단계) | LSTM과 같은 시퀀스 입력, 순환 셀만 GRU (`lstm.py`의 `arch: gru`) | LSTM과 같은 기본값 |
| GARCH(1,1) (2단계) | 종목별 종가 수익률, 분산 목표화 QMLE(학습 구간) → 다음 h일 평균 조건부 분산 | `mean: zero`, `min_obs: 250`(미만이면 다른 종목 추정치 중앙값, 처음 `warmup: 60`개 관측으로 분산), 결측일은 σ²로 갱신 |
| GNN (2단계) | 날짜별 그래프: 노드 = 그날 특성이 있는 universe 종목(구성종목 + 이벤트 종목), 엣지 = 최근 60일 수익률 상관 상위 k개. GraphSAGE 평균 집계 2층 | `k: 5`, `refresh: 5`일마다 상관 갱신, 은닉 32, 에포크 5 |
| TCN (3단계) | 같은 시퀀스 입력, 인과 팽창 합성곱(층 i의 팽창 2^i, 커널 3), 잔차 연결, 마지막 시점 출력 (Bai et al., 2018) | `lstm.py`의 `arch: tcn`, 채널 32, 3층 |
| Transformer (3단계) | 같은 시퀀스 입력, 선형 투영 + 학습 위치 임베딩 + 인코더, 마지막 시점 표현 (Vaswani et al., 2017) | `arch: transformer`, d_model 32, 헤드 2, 2층, 마스크 없음(창 전체가 t일 이전) |

- `rv ≤ 0`(가격이 하루 종일 같은 날 등) 행은 HAR에서 로그를 쓸 수 없어 예측하지 않는다. 공통 평가는 모든 모델이 예측한 행만
  쓰므로 이 행은 EWMA 평가에서도 빠진다.
- HAR은 로그에서 예측하므로 exp(ŷ)는 조건부 중앙값에 가깝다(평균보다 작음). QLIKE는 과소 예측에 벌점이 커서 불리할 수 있다.
  편향 보정(예: exp(ŷ + s²/2))은 하지 않았다 — 팀 논의 필요. LightGBM·LSTM도 로그 분산으로 학습하므로 같은 성질이 있다.
- LightGBM·LSTM은 학습 구간(`train`, `embargo == False`) 행만으로 학습하고, seed(`--seed`, 기본 0)와 설정·패키지 버전을 meta에 남긴다.
- 패키지는 `requirements-models.txt`에 따로 둔다 (CI·수집 환경에는 설치하지 않음, 해당 테스트는 패키지가 없으면 건너뜀):

```bash
.venv/Scripts/python -m pip install -r requirements-models.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

하이퍼파라미터는 검증 구간에서 튜닝한다 (`docs/TUNING.md`, 그리드 `configs/tuning.yaml`).
