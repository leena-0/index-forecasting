# 하이퍼파라미터 튜닝 (`src/tune`)

**튜닝·선택은 검증 구간(2022~2023)에서 끝내고, 테스트 구간은 마지막에 한 번만 본다** (CLAUDE.md 4.1, 10.3).
튜닝 코드는 테스트 구간 지표를 계산하지 않는다.

## 1. 실행

```bash
.venv/Scripts/python -m src.tune --model lgbm --h 5 --jobs 4     # 그리드: configs/tuning.yaml
```

| 결과 (git 제외) | 내용 |
|---|---|
| `data/tuning/{model}_h{h}/results.csv`, `results.md` | 설정 × 편향 보정별 검증 지표, `qlike_val` 순위 |
| `data/tuning/{model}_h{h}/val_preds/{config_id}.parquet` | 설정별 검증 구간 예측값 (`pred_log_var`, `pred_log_var_smeared`) |
| `data/preds/tuned/{모델}_h{h}.parquet` + `.meta.json` | 가장 좋은 설정의 전체 예측 테이블 (6.2 형식) — 공통 평가(`src.evaluate`)에 넣는다 |

## 2. 규칙

- 설정마다 학습 구간으로 학습하고, **편향 보정 없음/있음**을 둘 다 평가한다 (다시 학습하지 않음).
  편향 보정(smearing, Duan 1983): c = ln mean exp(y − ŷ) — 학습 구간 잔차만으로 추정 (`src/models/registry.py`).
- 지표 (분산 단위): `qlike_val`(주 지표)·`mse_val` = 검증 구간 평균, `qlike_event` = 검증 구간 편입 이벤트 윈도우 안 평균.
- 평가 행: 검증 구간, `embargo == False`, `yz_var > 0`, **그 모델의 모든 설정이 예측한 행** — 설정마다 표본이 같다.
  (입력 길이가 다른 설정이 섞이면 짧은 설정도 긴 설정의 표본으로 평가된다.)
- 선택: `qlike_val`이 가장 작은 (설정, 편향 보정). seed는 0으로 고정한다.
- 모델 사이 비교는 튜닝 표가 아니라 공통 평가(`src.evaluate`, 모든 모델 공통 표본)로 한다.
- 탐색 범위 (`configs/tuning.yaml`, 2026-11-05 변경): 2026-11-03 검증 튜닝에서 선택이 범위 끝에 걸린 축만 넓히고, 뚜렷하게 진 값은 뺐다.
  전체 조합을 다 넓히면 계산량이 지나치게 커서 축마다 값을 제한했다. 입력 길이 66이 들어가면 그 모델이 예측하는 행이 줄어 공통 표본도 작아질 수 있다.

## 3. 주 예측 기간 선택 (CLAUDE.md 7.4)

`src.evaluate`가 `horizon_choice.csv`를 만든다: 검증 구간 편입 이벤트 윈도우에서
imp_h = 평균_m (1 − L_m / L_EWMA), m ∈ {HAR, LightGBM, LSTM}. imp_h가 큰 h를 주 결과로 쓴다.
튜닝된 예측(`data/preds/tuned/`)으로 실행한다.

## 4. 실행 기록 (개발용 — 결과 아님)

두 실행은 타깃 정의가 달라 섞어 비교하지 않는다. 경로는 각자 로컬의 `data/` 아래다 (git 제외).

### 4.1 2026-11-06 재튜닝 (기준가 규칙)

- 조건: 코드 `b7bd9e3`(기준가 규칙·거래정지 직후 제외·주식수 시차 보정, 범위 #41), 구성종목·이벤트 legacy, 공지 없는 이벤트 16건 제외, `reportable: false`
- 범위: 9개 모델 × h 2개, `--jobs 3`으로 약 6시간 53분, 실패 0. 결과 `data/tuning_bp/`, 고른 예측 `data/preds/tuned_bp/`, 검증 평가 `data/eval/tuned_val_bp/`
- 선택: 18개 모두 편향 보정 있음. 굵게 쓴 값은 탐색 범위 끝이다. 후보가 2개인 축(GNN 전 축, LSTM 전 축 등)은 어느 값을 골라도 끝이므로,
  후보가 3개 이상인 축에서 끝에 걸린 것(Transformer lookback 66, TCN h = 22 lookback 22, LightGBM 트리 100, GARCH min_obs 120, GRU h = 22 에포크 5)이 범위 확장 판단에 더 의미가 있다.

| 모델 | h = 5 | h = 22 |
|---|---|---|
| EWMA | λ 0.97 | λ 0.99 |
| HAR | (설정 없음) | (설정 없음) |
| GARCH | mean demean, min_obs **120** | mean demean, min_obs **120** |
| LightGBM | lr **0.03**, 잎 15, 최소 1000, 트리 **100** | lr **0.03**, 잎 15, 최소 1000, 트리 **100** |
| LSTM | lookback **66**, 은닉 **16**, 에포크 **10** | lookback **66**, 은닉 **8**, 에포크 **20** |
| GRU | lookback **66**, 은닉 **16**, 에포크 10 | lookback **66**, 은닉 **16**, 에포크 **5** |
| GNN | k **5**, 은닉 **64**, 에포크 **5** | k **3**, 은닉 **32**, 에포크 **3** |
| TCN | lookback 44, 은닉 **32**, 에포크 **5** | lookback **22**, 은닉 **32**, 에포크 **5** |
| Transformer | lookback **66**, 은닉 **16**, 층 **2** | lookback **66**, 은닉 **32**, 층 **3** |

- 공통 평가 표본 (검증): h = 5 138,426행, h = 22 132,739행. lookback 66 때문에 모델별 최대 표본(169,531·162,992행)보다 약 18% 작다.
- 검증 구간 전체 MCS(QLIKE, α = 0.10): h = 5 {GRU, Transformer}, h = 22 {LSTM, GRU, Transformer}. 이벤트 윈도우 평균 QLIKE는 h = 5 LightGBM, h = 22 GNN이 가장 낮다 (이벤트 윈도우만의 검정은 아직 없음, `docs/EVALUATION.md` 3절).
- 7.4: h = 22 선택 (개선율 29.9%, h = 5는 27.9%). 모델별 값은 CLAUDE.md 7.4.
- 범위를 다시 넓힐지는 팀 확인 뒤 정한다. lookback을 더 늘리면 공통 표본이 더 줄어든다.

### 4.2 2026-11-03 튜닝 (이전 5.4 규칙)

- 조건: 구성종목·이벤트 legacy, 공지 없는 이벤트 16건 제외, 이전 5.4 규칙(`corp_action` 결측), `reportable: false`
- 범위: 9개 모델 × h 2개, 약 4시간. 결과 요약은 `data/tuning/SUMMARY.md`
- 선택: 18개 모두 편향 보정 있음. 여러 선택이 탐색 범위 끝에 걸렸다(lookback 44, epochs 10, λ 0.99) → 범위를 넓혔다 (#41, 2절).
- 7.4: h = 22 선택 (개선율 31.0%, h = 5는 25.7%, `data/eval/tuned_val`)
