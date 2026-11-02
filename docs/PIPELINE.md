# 파이프라인 사용법·설정 (`src/pipeline`)

1단계(데이터 → 타깃 → 분할·경계 공백 → 이벤트 구간). 모델은 아직 없다.
처음 설정(가상환경, KRX Open API 인증키, `.env`)은 **`docs/SETUP.md`**를 먼저 본다.

> ⚠️ 구성종목·이벤트는 아직 **개발용 기존 수집 데이터**(pykrx 자동 수집)에서 온다.
> 가격은 `dev_legacy`(네이버 수정주가) 또는 `openapi`(KRX Open API, 공식 경로 — 8절) 중 고른다.
> 어느 쪽이든 이 상태의 수치는 결과로 보고하지 않는다 (CLAUDE.md 10.2).

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
| `meta.json` | data_source, as_of, 사용한 config 전체, git commit, 패키지 버전, 건수 요약, **`reportable: false`와 `hold_reasons`** (구성종목·이벤트가 legacy이고 공지일이 임시값이라 최종 결과에 반영하지 않음) |

모델이 쓸 수 있는 예측 행 = `yz_var`가 결측이 아니고 `embargo == False`.
이벤트 구간 분석에 쓸 수 있는 행 = `complete == True` 이고 `crosses_split == False`.

테스트: `.venv/Scripts/python -m unittest` (합성 데이터, 네트워크 없음)

## 2. 설정 (`configs/pipeline.yaml`)

| 키 | 기본값 | 설명 |
|---|---|---|
| `data_source` | `openapi` | 데이터 어댑터. `dev_legacy`(기존 데이터만) / `openapi`(가격·지수는 Open API, 구성종목·이벤트는 기존 데이터) |
| `legacy_dir` | `null` | 기존 데이터 폴더. `null`이면 환경변수 `LEGACY_DATA_DIR`. `openapi`에서도 구성종목·이벤트용으로 필요 |
| `openapi_dir` | `data/raw/openapi` | Open API 수집 폴더 (`openapi`일 때) |
| `corporate_action.gap_limit` | `0.30` | 수정 안 된 가격에서 가격 기준 변경일 판정: ratio = 시가/전일 종가가 ratio > 1 + 값 또는 ratio < 1 − 값 (기본 1.3 / 0.7) — 8절 |
| `corporate_action.shares_ratio` | `1.5` | 같은 판정: 상장주식수가 전일 대비 이 배수 이상(또는 역수 이하)으로 바뀐 날 |
| `as_of` | `"2024-12-30"` | 데이터 기준일. 이후 데이터는 잘라낸다. 정식 값은 2025년 마지막 거래일이지만 기존 데이터가 2024-12-30까지다 |
| `split.train_end` | `"2022-12-31"` | 학습 구간 끝 (CLAUDE.md 4.1) |
| `split.val_end` | `"2023-12-31"` | 검증 구간 끝. 테스트 = 그다음 날 ~ `as_of` |
| `horizons` | `[5, 22]` | 예측 기간 h(영업일). 분할 경계마다 h일씩 비운다 |
| `yang_zhang.demean` | `true` | Yang–Zhang 분산의 평균 처리 — 아래 3절 |
| `event_phases` | CLAUDE.md 4.3 | 구간별 기준점(`anchor`: `announce` 공지일 / `effective` 편입일)과 영업일 오프셋 `range: [시작, 끝]`(양 끝 포함). `announce_to_effective`(공지일 ~ 편입일 직전)는 고정 |
| `placeholder_announce` | `effective` | 공지일이 없을 때 쓰는 임시 공지일. 지금은 편입일만 지원 — 아래 4절 |
| `events_path` | `data/raw/events/announcements.csv` | 실제 공지일 CSV (`python -m src.events from-notices`로 생성). 설정하면 기존 구성종목 변화와 일대일 대조한 뒤 실제 공지일을 사용한다. 형식은 `docs/EVENTS.md`. `null`이면 임시 공지일 |
| `events_missing` | `drop` | 공지일이 없는 이벤트 처리. `error` = 중단(일대일 대응 필수), `drop` = 빼고 진행(건수는 출력·meta·`hold_reasons`에 남김). 구성종목 변화가 없는 공지는 어느 쪽이든 중단 |
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

→ 이 상태의 구간별 결과는 해석하지 않는다. 실제 공지일을 구축한 뒤 `events_path`에
`data/raw/events/announcements.csv`를 설정하면 `src/events`가 기존 구성종목 변화와 일대일 대조하고,
실제 공지일과 `announce_source`의 `crawled`/`manual` 값을 사용한다.

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
5. 구성종목 이력·이벤트는 `openapi`에서도 기존 데이터(2024-12-30까지)를 쓴다. 공식 출처(DECISIONS.md 3.3)가 정해지면 교체한다.

## 8. KRX Open API 수집·어댑터 (`src/krxapi`, `src/pipeline/openapi.py`)

공식 경로. 승인된 API 세 개를 쓴다: **코스닥 일별매매정보**(`sto/ksq_bydd_trd`, 날짜별 코스닥 전 종목),
**KOSDAQ 시리즈 일별시세정보**(`idx/kosdaq_dd_trd`, `IDX_NM == "코스닥 150"` 행), **코스닥 종목기본정보**
(`sto/ksq_isu_base_info`, 상장일). 유가증권 일별매매정보는 신청하지 않았다. 명세서: `docs/API 명세서/Spec.docx`.

```bash
# .env: KRX_OPENAPI_KEY(인증키), COLLECTOR_ID
.venv/Scripts/python -m src.krxapi collect --date 2015-07-13  # 시험: 하루만
.venv/Scripts/python -m src.krxapi collect --limit 5          # 시험: 처음 5평일
.venv/Scripts/python -m src.krxapi collect                    # 전체 (하루 한도에 닿으면 멈춤 → 다음 날 다시 실행하면 이어 받음)
.venv/Scripts/python -m src.krxapi validate --complete        # 저장된 파일 검사 (네트워크 없음)
.venv/Scripts/python -m src.krxapi verify                     # MANIFEST 해시 대조
```

- 설정 `configs/krxapi.yaml`: `as_of`(2025-12-30, [미확인]), `start`, `sleep_sec`(0.5), `retries`(3), `daily_limit`(9,500 — 키당 하루 10,000건)
- 순서: 평일마다 지수 → `코스닥 150` 행이 있는 날(거래일)마다 종목 → 달마다 마지막 거래일에 종목기본정보.
  - **휴장일 판정**: 지수가 비면 종목 API도 비었는지 확인해야 휴장일로 본다. 휴장일은 0행 지수·종목 파일로 남겨 다시 요청하지 않는다.
    지수만 비고 종목 데이터가 있으면 실패로 기록한다. 끝나면 연도별 평일 휴장 수를 출력한다(KRX 휴장일과 대조).
  - 종목기본정보는 과거 날짜로 조회하면 그날 상장돼 있던 종목 전체(나중에 상장폐지된 종목 포함)가 오고 상장일은 바뀌지 않으므로
    월 1회면 충분하다 (한 달 안에 상장·폐지된 종목만 빠질 수 있음).
- **응답 검증** (실패하면 파일을 만들지 않고, 다시 실행하면 그 단위만 받는다): 필수 열, `BAS_DD == 요청일`,
  종목코드(`ISU_CD`, `ISU_SRT_CD`) 빈 값·중복, `코스닥 150` 행이 정확히 1개, 거래일인데 0행.
- **`validate`**: 휴장일 표시 파일 짝, 거래일 종목 파일 존재·검증, 거래가 있던 종목의 OHLC 관계(저가 ≤ 시가·종가 ≤ 고가,
  거래정지 행 제외), 월말 종목기본 파일. `--complete`는 모든 평일 지수 파일이 있는지도 본다.
- **호출 수**: `data/_state/openapi-requests-YYYY-MM-DD.json`에 날짜별로 누적한다. 같은 날 여러 번 실행해도 합산해 `daily_limit`에서 멈춘다.
  429(요청 제한)도 그날 수집을 멈춘다. 종료 코드: 0 정상 / 1 실패한 단위 있음 / 2 한도·요청 제한으로 멈춤.
- 401 `Unauthorized Key` = 키 문제, `Unauthorized API Call` = 그 API 이용신청이 승인되지 않음 → 재시도하지 않고 멈춘다.
  네트워크 오류·5xx·JSON이 아닌 응답은 최대 `retries`회 재시도한다.
- 저장: `data/raw/openapi/{kosdaq_index|kosdaq_daily|kosdaq_basic}/date=YYYYMMDD.parquet` (원본 문자열), `progress.jsonl`, `MANIFEST.csv`, `snapshot.json`
- 요청 수: 2015-01 ~ 2025-12 평일 약 2,870 + 거래일 약 2,710 + 휴장일 확인 약 160 + 월말 132 ≈ 5,900건 (하루 안에 가능)
- 휴장일 이중 확인, 응답 검증, `validate`, 날짜별 호출 수 기록, 429 처리, `--date`는 이나영 님 수집기(`topic/krx-openapi`의 `src/crawl`)에서 가져왔다.

**어댑터에서 처리하는 것**

- 확인된 사실 (2015-07-13 시험): `ISU_CD`는 6자리 단축코드, 숫자에 쉼표 없음, **수정주가가 아님**(카카오 종가 134,800),
  그날 KOSDAQ150 150종목이 모두 응답에 있음(현재 상장폐지된 16종목 포함 → 생존 편향 없음), 코스닥 150 지수 값이 기존 데이터와 같음.
- 숫자 변환: 쉼표 제거, `""`·`"-"`는 결측. 거래정지(시가·고가·저가 0·결측 또는 거래량 0)는 OHLC·거래량 결측.
- **가격 기준 변경일(`corp_action`)**: 가격이 수정되지 않았으므로 분할·무상증자 등으로 기준가가 바뀐 날의 야간 수익률은 실제 수익률이 아니다.
  시가/전일 종가 비율이 1.3 초과 또는 0.7 미만(가격제한폭 ±30% 밖)이거나 상장주식수가 1.5배 이상 바뀐 날을 표시하고, 그날이 들어간 타깃 구간은 결측으로 둔다
  (DECISIONS.md 3.2의 "이벤트일 제외" 안을 보수적으로 적용). Yang–Zhang의 장중 항은 같은 날 가격 비율이라 영향이 없다.
  한계: 가격제한폭 안의 작은 기준가 조정(소규모 유상증자 권리락·배당락)은 잡지 못한다. 2015-06-15 이전 가격제한폭은 ±15%였다.
- **이전 상장 종목**: 승인된 API는 코스닥뿐이라, 유가증권시장으로 옮긴 종목은 이전 이후 데이터가 없다. 그 구간은
  `event_phases`에서 "종목 데이터 기간 밖"으로 자동 제외되고, `prepare`가 이전 상장 추정 종목 수와 제외된 구간 수를 출력한다.
  판정(`early_end`): universe 종목 중 코스닥 데이터가 as_of 전에 끝나는데 기존 데이터는 7일 넘게 이어지면 `moved`, 같이 끝나면 `ended`(상장폐지 추정).
- 상장일(`listing`): 월말 종목기본정보의 `LIST_DD`를 종목별로 모은다.
