# 공통 평가 사용법 (`src/evaluate`)

모델마다 평가 코드를 따로 만들지 않는다 (CLAUDE.md 6.2). 모든 모델은 예측 테이블만 내고, 이 코드가 지표를 계산한다.

> ⚠️ prepare 결과의 `reportable: false`가 평가 결과 `meta.json`에도 그대로 넘어간다. 이 상태의 수치는 결과로 보고하지 않는다.

## 1. 실행

```bash
# 먼저 prepare 결과 폴더가 있어야 한다 (members.parquet 포함 — 예전 결과 폴더는 prepare를 다시 실행)
.venv/Scripts/python -m src.evaluate --pred 예측1.parquet 예측2.parquet --data-dir data/dev --split val
```

- `--split test`는 `--final`을 함께 줘야 실행된다. 테스트 구간은 한 번만 본다 (CLAUDE.md 4.1, 열람 기록도 여기에 있다).
- `--alpha`(MCS 유의수준, 기본 0.10), `--B`(부트스트랩 횟수, 기본 1000), `--seed`(기본 0), `--out`(기본 `data/eval/<split>`)

예측 테이블 열 (CLAUDE.md 6.2): `ticker, origin_date, horizon(5|22), pred_log_var, model, seed, version`

| 결과 파일 | 내용 |
|---|---|
| `summary.csv` | split, horizon, model별 평가 행 수와 평균 QLIKE·MSE |
| `dm_h{h}.csv` | 모델 쌍별 Diebold–Mariano (QLIKE). `stat < 0`이면 `model_1`의 손실이 작다 |
| `mcs_h{h}.csv` | Model Confidence Set (QLIKE). `in_mcs`, `mcs_p`, 빠진 순서 |
| `sensitivity.csv` | 편입 민감도 Sens_{m,s} (QLIKE), 구간별 이벤트 수 |
| `sensitivity_all_baselines.csv` | 기준 구간 겹침도 포함한 민감도 (이전 방식과 비교) |
| `sensitivity_no_prior_event_overlap.csv` | 다른 종목의 이전 이벤트와 날짜가 겹친 기준 구간까지 제외한 민감도 (표본 감소 확인용) |
| `baseline_overlap_audit.csv` | 편입 이벤트별 기준 구간 겹침·완전성·분할 경계 여부 |
| `no_baseline_post.csv` | 유효한 기준 구간이 없는 편입 이벤트의 사후 구간 ln(L^E / L^C) |
| `event_window.csv` | 편입 이벤트 윈도우 안의 평균 QLIKE·MSE (split, horizon, model) |
| `horizon_choice.csv` | 7.4 주 예측 기간 선택: EWMA 대비 1단계 모델 평균 QLIKE 개선율 (`docs/TUNING.md` 3절). 7.4는 검증 구간 규칙이라 `--split val`의 파일만 쓴다 (테스트 평가에도 만들어지지만 쓰지 않는다) |
| `meta.json` | 입력 모델·seed·version, 설정, 건수, prepare의 `reportable`·`hold_reasons` |

## 2. 계산 방식과 가정

정의식은 각 모듈의 docstring에 있다 (`losses.py`, `compare.py`, `sensitivity.py`).

1. **공통 표본**: 모든 모델이 예측한 (종목, 날짜, h)만 평가한다. 빠진 행 수는 `meta.json`의 `dropped_not_common` (CLAUDE.md 6.3).
2. **제외 행**: `embargo == True`, 타깃 결측, `yz_var ≤ 0`(QLIKE 계산 불가).
3. **seed가 여러 개**: 예측값을 평균하지 않고 seed별 손실을 평균한다.
4. **DM·MCS 입력**: 날짜마다 종목 평균 손실을 낸 시계열. DM은 Newey–West HAC(시차 h−1), MCS는 T_max 통계량 +
   원형 블록 부트스트랩(블록 길이 h).
5. **대조군**: 그날의 구성종목 중 어떤 편입·편출 이벤트 윈도우(첫 구간 시작 ~ 마지막 구간 끝)에도 들지 않은 종목.
6. **L^E, L^C**: 이벤트 × 날짜를 모두 모은 평균. 구간 s마다 기준 구간과 s가 모두 완전한 편입 이벤트만 쓴다.
   공통 표본에서 기준 구간의 예측이 하나도 없거나 같은 종목의 이전 이벤트 단기 충격·사후 구간과 겹친 이벤트는 `sensitivity.csv`에서 제외한다. 다른 종목의 이전 이벤트와 같은 날짜가 겹치는 경우는 별도 결과로 비교한다. 겹침 자체는 구간의 `complete` 값을 바꾸지 않는다.
7. **기준 구간 없는 사후 비교**: 기준 구간 불완전·예측 결측·같은 종목 겹침으로 기준 구간을 사용할 수 없는 이벤트는 `no_baseline_post.csv`의 사후 비교 후보가 된다. 2026-11-06 기존 검증 평가(겹침 규칙 적용 전, 개발용)에서는 편입 이벤트 25건 중 8건이 해당했다 — 기준 구간 불완전 3건, 시퀀스 모델 66일 입력 창 안 결측 4건, 기준 구간 타깃 결측 1건. 새 겹침 규칙의 건수는 재실행 후 `meta.json`의 `counts.baseline_overlap`에서 확인한다. 기존 prepare 산출물에는 새 겹침 열이 없어 다시 prepare해야 한다.
8. **편입 민감도 해석**: Sens는 시장 공통 변화(대조군 손실 변화)를 차감한 편입 관련 **상대적 오차 변화**다. 인과효과로 해석하지 않는다
   (편입은 시가총액 순위로 정해지는 비무작위 선정이고, 평행 추세를 확인하지 않음 — CLAUDE.md 7.2). 모델 간 상대 비교에 쓴다.

이 겹침 규칙은 편입 민감도에만 적용한다. 전체 구간 QLIKE·MSE, DM·MCS, 이벤트 윈도우 평균, 예측 기간 선택 계산은 그대로다.

## 3. 아직 없는 것

- (7.4 주 예측 기간 선택은 `horizon_choice.csv`로 추가됨)
- 이벤트 윈도우만 떼어 낸 DM·MCS. 지금은 분할 전체로 검정한다.
- 편입 민감도의 불확실성(예: 이벤트 단위 부트스트랩 신뢰구간). 지금은 점추정만 낸다.
