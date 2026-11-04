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

## 3. 주 예측 기간 선택 (CLAUDE.md 7.4)

`src.evaluate`가 `horizon_choice.csv`를 만든다: 검증 구간 편입 이벤트 윈도우에서
imp_h = 평균_m (1 − L_m / L_EWMA), m ∈ {HAR, LightGBM, LSTM}. imp_h가 큰 h를 주 결과로 쓴다.
튜닝된 예측(`data/preds/tuned/`)으로 실행한다.

## 4. 실행 기록 (2026-11-03, 개발용 — 결과 아님)

- 조건: 구성종목·이벤트 legacy, 공지 없는 이벤트 16건 제외, 이전 5.4 규칙(`corp_action` 결측), `reportable: false`
- 범위: 9개 모델 × h 2개, 약 4시간. 결과 요약은 `data/tuning/SUMMARY.md`
- 선택: 18개 모두 편향 보정 있음. 여러 선택이 탐색 범위 끝에 걸렸다(lookback 44, epochs 10, λ 0.99) → 다시 튜닝하기 전에 `configs/tuning.yaml`의 범위 확장을 검토한다.
- 7.4: h = 22 선택 (개선율 31.0%, h = 5는 25.7%, `data/eval/tuned_val`)
- 기준가 규칙(CLAUDE.md 5.4)을 코드에 반영하면 타깃이 바뀌므로 prepare·튜닝·7.4를 다시 실행하고, 이 결과와 섞어 비교하지 않는다.
