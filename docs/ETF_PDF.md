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
```

Windows는 `.venv/bin/python` 대신 `.venv/Scripts/python`을 사용한다.

출력은 `data/interim/etf_pdf/`에 생성된다.

| 파일 | 내용 |
|---|---|
| `monthly_equities.parquet` | 월말 주식 구성종목과 원본·정규화 비중 |
| `monthly_cash.parquet` | 원화예금/원화현금 행 |
| `snapshots.parquet` | 파일별 날짜·해시·행 수·비중합 |
| `meta.json` | 기간, 파일 수, 형식, 용도 및 정규화 규칙 |

주식 종목코드는 6자리 문자열로 보정한다. `raw_weight_pct`는 원본 값을 보존하고,
`equity_weight_pct`는 현금을 제외한 주식끼리 합계 100%가 되도록 정규화한다.
`source_sha256`으로 사용한 원본을 추적한다.

## 아직 필요한 자료

123개는 월말 스냅샷이다. 정기변경 적용일과 수시 편입·편출일을 정확하게 반영하려면 해당 적용일의 PDF를
추가로 내려받아야 한다. 월말 사이의 일별 비중 복원은 Open API 가격 데이터와 결합하는 별도 단계에서 구현한다.
