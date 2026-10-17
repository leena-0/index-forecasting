# 파이프라인 사용법·설정 (`src/pipeline`)

브랜치: `topic/pipeline-dev` · 1단계(데이터 → 타깃 → 분할·경계 공백 → 이벤트 구간). 모델은 아직 없다.

> ⚠️ 지금 데이터는 **개발용 기존 수집 데이터**(`data_source: dev_legacy`, pykrx·네이버 자동 수집)다.
> 이 데이터로 나온 수치는 결과로 보고하지 않는다 (CLAUDE.md 10.2). 공식 경로 데이터가 생기면
> 같은 형식을 돌려주는 어댑터를 추가하고 `data_source`만 바꾼다.

## 1. 실행

```bash
# 기존 데이터 폴더 지정 — 둘 중 하나
#   (a) configs/pipeline.yaml의 legacy_dir에 경로 적기 (로컬 경로라 커밋하지 않는 것을 권장)
#   (b) .env 또는 환경변수에 LEGACY_DATA_DIR=.../kosdaq150_forecast/data
.venv/Scripts/python -m src.pipeline prepare                 # Windows
python -m src.pipeline prepare --config configs/pipeline.yaml  # 설정 파일 지정
```

결과는 `data/dev/`(git 제외)에 저장된다.

| 파일 | 내용 |
|---|---|
| `targets_h{h}.parquet` | `ticker, origin_date, horizon, yz_var, split, embargo` — 종목·거래일마다 다음 h일의 Yang–Zhang 분산 |
| `event_phases.parquet` | 이벤트 × 5구간: `start, end, complete, reason, split, crosses_split, baseline_overlaps_prev` |
| `meta.json` | data_source, as_of, 사용한 config 전체, git commit, 패키지 버전, 건수 요약 |

모델이 쓸 수 있는 예측 행 = `yz_var`가 결측이 아니고 `embargo == False`.
이벤트 구간 분석에 쓸 수 있는 행 = `complete == True` 이고 `crosses_split == False`.

테스트: `.venv/Scripts/python -m unittest` (합성 데이터, 네트워크 없음)

## 2. 설정 (`configs/pipeline.yaml`)

| 키 | 기본값 | 설명 |
|---|---|---|
| `data_source` | `dev_legacy` | 데이터 어댑터. 지금은 `dev_legacy`만 있다 |
| `legacy_dir` | `null` | 기존 데이터 폴더. `null`이면 환경변수 `LEGACY_DATA_DIR` |
| `as_of` | `"2024-12-30"` | 데이터 기준일. 이후 데이터는 잘라낸다. 정식 값은 2025년 마지막 거래일이지만 기존 데이터가 2024-12-30까지다 |
| `split.train_end` | `"2022-12-31"` | 학습 구간 끝 (CLAUDE.md 4.1) |
| `split.val_end` | `"2023-12-31"` | 검증 구간 끝. 테스트 = 그다음 날 ~ `as_of` |
| `horizons` | `[5, 22]` | 예측 기간 h(영업일). 분할 경계마다 h일씩 비운다 |
| `yang_zhang.demean` | `true` | Yang–Zhang 분산의 평균 처리 — 아래 3절 |
| `event_phases` | CLAUDE.md 4.3 | 구간별 기준점(`anchor`: `announce` 공지일 / `effective` 편입일)과 영업일 오프셋 `range: [시작, 끝]`(양 끝 포함). `announce_to_effective`(공지일 ~ 편입일 직전)는 고정 |
| `placeholder_announce` | `effective` | 공지일이 없을 때 쓰는 임시 공지일. 지금은 편입일만 지원 — 아래 4절 |
| `out_dir` | `data/dev` | 결과 저장 폴더 (레포 루트 기준) |

### 3. `yang_zhang.demean` — n = 5에서의 평균 처리 (CLAUDE.md 5.3)

h일 구간(n = h)의 야간 수익률 o, 장중 수익률 c의 분산을 어떻게 구할지 정한다.

| 값 | 식 | 특징 |
|---|---|---|
| `true` (기본) | σo² = 1/(n−1)·Σ(o − ō)², σc² = 1/(n−1)·Σ(c − c̄)² | Yang & Zhang (2000) 원래 정의. n = 5에서 자유도 4라 흔들릴 수 있다 |
| `false` | σo² = 1/n·Σo², σc² = 1/n·Σc² | 평균 0 가정. 짧은 구간에서 덜 흔들리지만 추세가 있으면 커진다 |

두 경우 모두 σRS² = 1/n·Σ[u(u−c) + d(d−c)], σYZ² = σo² + k·σc² + (1−k)·σRS²,
k = 0.34 / (1.34 + (n+1)/(n−1)) (n = 5 → 0.120, n = 22 → 0.140). 식은 `src/pipeline/target.py` docstring에도 있다.
값을 바꾸면 이전 결과와 비교할 수 없으므로 `meta.json`의 `config`로 어느 설정인지 확인한다.

### 4. 임시 공지일 (`placeholder_announce`)

시점 규칙(CLAUDE.md 3장)과 5구간(4.3)은 공지일 기준인데, 기존 데이터에는 공지일이 없다.
개발용으로 **편입일을 공지일로 쓰고 `announce_source = placeholder`** 로 표시한다. 그 결과:

- `announce_to_effective` 구간은 항상 빈 구간 (`reason = 빈 구간 (임시 공지일)`)
- `baseline`·`anticipation`은 실제보다 공지~편입 기간만큼 뒤로 밀린다
- 이벤트의 분할 배정도 실제 공지일과 다를 수 있다

→ 이 상태의 구간별 결과는 해석하지 않는다. 공지일 모듈(`src/events/`, CLAUDE.md 2.4)이 생기면
`announcements.csv`를 읽어 실제 공지일을 넣고 `announce_source`를 `crawled`/`manual`로 바꾼다.

## 5. 처리 규칙 요약

- **거래정지**: 시가·고가·저가가 0이거나 **거래량이 0인 날**은 OHLC·거래량을 결측으로 둔다 (보간 없음).
  기존 데이터에는 OHLC가 직전 종가로 채워지고 거래량만 0인 날이 있어 거래량 0도 거래정지로 본다.
- **타깃 결측**: 구간 안에 결측이 하나라도 있으면 NaN. 거래정지 다음 날은 직전 종가가 없어 그날이 들어간 구간도 NaN.
  상장 전·데이터 끝(as_of 이후)도 NaN.
- **경계 공백**: origin의 분할과 타깃 구간 끝(origin + h영업일)의 분할이 다르면 `embargo = True`.
- **이벤트 분할**: 공지일 기준. 전체 윈도우(기준 구간 시작 ~ 사후 구간 끝)가 경계를 넘으면 `crosses_split = True`.
- **불완전 구간**: 달력 시작 전 / as_of 이후 / 종목 데이터 기간 밖이면 `complete = False`, 이유는 `reason`.
- **기준 구간 겹침**: 같은 종목의 직전 이벤트 사후 구간과 겹치면 `baseline_overlaps_prev = True` (표시만).
- **구성종목 검증**: 편입 이벤트는 편입일에 구성종목이고 직전 거래일엔 아닌지(편출은 반대) 확인해 불일치 건수를 출력한다.

## 6. 기존 데이터로 실행한 결과 (2026-10-17, 개발용 — 결과 아님)

`prepare` 출력의 건수만 기록한다 (파이프라인 점검용).

- 거래일 2,328 (2015-07-13 ~ 2024-12-30), 종목 386, 이벤트 584 (구성종목 변화와 불일치 0)
- h = 5: 사용 가능 예측 행 train 601,401 / val 86,562 / test 84,517, yz_var ≤ 0 3건 (5일 내내 가격이 같은 구간, −8e−8 수준 부동소수점 오차)
- h = 22: train 591,353 / val 80,159 / test 78,225, yz_var ≤ 0 0건
- 이벤트(공지일 기준 분할): train 462 / val 64 / test 58. **윈도우가 분할 경계를 넘어 제외: train 28 / val 64(전부) / test 38**

## 7. 알려진 문제 · 팀 논의 필요

1. **검증 구간에 쓸 수 있는 이벤트가 없다.** 이벤트 윈도우(공지일 −120 ~ 편입일 +120 ≈ 240영업일)가
   검증 구간 1년(2023, 약 245영업일)과 길이가 비슷해, "윈도우가 경계를 넘으면 그 구간에서 제외"(CLAUDE.md 3장)를
   적용하면 6월·12월 정기변경 이벤트는 모두 빠진다. 그러면 주 예측 기간 선택 규칙(7.4, 검증 구간 이벤트 윈도우 기준)을
   쓸 수 없다. 규칙은 [확정]이라 코드는 그대로 두었고, 대안(구간별로 경계 판정, 검증 구간 연장 등)은 팀 논의가 필요하다.
2. **as_of와 테스트 구간**: 기존 데이터에 2025년이 없어 테스트 구간이 2024년뿐이다. 테스트 구간은 개발 중에 보지 않는다(4.1).
3. **yz_var ≤ 0**: 모델은 로그 분산으로 학습하므로(5.2) 0 이하 값의 처리(제외 등)를 모델 단계에서 정해야 한다. 건수는 `meta.json`에 있다.
4. 견고성 확인용 편입일 ±250영업일 윈도우(4.3)와 종가-종가 실현변동성 타깃(5.2)은 아직 없다.
