# ETF PDF 월말 비중 데이터

KODEX 코스닥150(229200)의 PDF(Portfolio Deposit File)를 KOSDAQ150 구성 비중의 **대용치(proxy)** 로 사용한다.
이는 KRX가 발표한 공식 지수 구성 비중이 아니라 ETF 보유내역이라는 한계가 있으므로 결과와 보고서에 구분해 적는다.

## 원본

- 기간: 2015-10 ~ 2025-12 월말, 총 123개
- 파일명: `229200_YYYYMMDD.xlsx` 또는 `229200_YYYYMMDD.csv`
- XLSX: 2015-10 ~ 2021-12, 75개
- CSV(CP949): 2022-01 ~ 2025-12, 48개
- 로컬 위치: `data/manual/etf_pdf/`
- 원본은 수정하거나 형식을 변환하지 않는다. `data/`는 Git에서 제외한다.

여기서 PDF는 문서 파일 확장자가 아니라 Portfolio Deposit File을 뜻한다.

## 실행

```bash
# 월 누락·중복, 필수 열, 종목코드, 현금 행, 비중합 검사
.venv/bin/python -m src.etf_pdf validate

# 검사 후 표준 Parquet 생성
.venv/bin/python -m src.etf_pdf prepare

# Open API 종목 데이터와 결합해 월말 사이의 일별 비중 복원
.venv/bin/python -m src.etf_pdf build-daily
```

Windows는 `.venv/bin/python` 대신 `.venv/Scripts/python`을 사용한다.

출력은 `data/interim/etf_pdf/`에 생성된다.

| 파일 | 내용 |
|---|---|
| `monthly_equities.parquet` | 월말 주식 구성종목과 원본·정규화 비중 |
| `monthly_cash.parquet` | 원화예금/원화현금 행 |
| `snapshots.parquet` | 파일별 날짜·해시·행 수·비중합 |
| `meta.json` | 기간, 파일 수, 형식, 용도 및 정규화 규칙 |
| `daily_weights.parquet` | 스냅샷 재설정과 스냅샷 사이 drift를 반영한 일별 종목 비중 |
| `daily_weight_quality.parquet` | 날짜별 완전성, 결측 종목 수, 비중합 |
| `daily_weights_meta.json` | drift 방법, 기간, 완전·불완전 날짜 수 |

주식 종목코드는 6자리 문자열로 보정한다. `raw_weight_pct`는 원본 값을 보존하고,
`equity_weight_pct`는 현금을 제외한 주식끼리 합계 100%가 되도록 정규화한다.
`source_sha256`으로 사용한 원본을 추적한다.

## 일별 비중 복원

스냅샷 날짜에는 `equity_weight_pct`로 비중을 재설정하고, 다음 스냅샷 직전까지는
`w(i,t) ∝ w(i,s) × value(i,t) / value(i,s)`로 drift한 뒤 합계 100%로 정규화한다.
기본 `value`는 Open API `MKTCAP`이다. 수정되지 않은 종가보다 분할·병합에 강하지만,
유상증자 같은 상장주식 수 변화에도 영향을 받으므로 정확한 ETF 보유 비중이 아닌 대용치다.
비교가 필요하면 `--method close`를 사용할 수 있지만 기업행사에 따른 가짜 가격 점프에 주의한다.

구성종목 값이 하나라도 결측이거나 0 이하인 날짜는 남은 종목만으로 재정규화하지 않는다.
그 날짜 전체를 `complete=False`와 비중 결측으로 남겨 조용한 표본 왜곡을 막는다.

`build-daily`는 최신 팀원 수집기의 `data/raw/openapi/kosdaq_daily/date=YYYYMMDD.parquet`
구조를 사용한다. 예전 `stock_daily/year=...` 자료와 자동으로 섞지 않는다.

## 아직 필요한 자료

123개는 월말 스냅샷이다. 정기변경 적용일과 수시 편입·편출일을 정확하게 반영하려면 해당 적용일의 PDF를
추가로 내려받아야 한다. 해당 파일이 표준 스냅샷에 추가되면 그 날짜부터 비중을 다시 설정하도록 연결한다.
