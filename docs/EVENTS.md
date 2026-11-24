# KOSDAQ150 편입·편출 이벤트 데이터

정기변경, 수시변경, 특례편입의 공지일과 실제 적용일을 공통 CSV로 관리한다.
출처가 자동 수집을 허용하지 않으면 수동으로 작성하고 `source_type=manual`로 표시한다.
API 키와 원문 파일은 Git에 올리지 않으며, 이 CSV도 `data/` 아래에 있어 기본적으로 Git에서 제외된다.

## 파일 형식

기본 위치: `data/raw/events/announcements.csv`

| 열 | 값 |
|---|---|
| `ticker` | 종목코드. 읽을 때 6자리로 맞춘다. |
| `event_type` | `정기변경`, `수시변경`, `특례편입` |
| `direction` | `편입`, `편출` |
| `announce_date` | 공지일 `YYYY-MM-DD` |
| `effective_date` | 실제 적용일 `YYYY-MM-DD` |
| `source_url` | 공지 원문 URL |
| `source_type` | `manual` 또는 `crawled` |
| `collected_at` | 수집·확인 일시 |

예시:

```csv
ticker,event_type,direction,announce_date,effective_date,source_url,source_type,collected_at
1234,정기변경,편입,2024-05-24,2024-06-14,https://example.com/notice/1,manual,2026-10-28T10:00:00+09:00
```

## 검사

```bash
.venv/Scripts/python -m src.events validate   # macOS·Linux는 .venv/bin/python
```

다음을 검사한다.

- 필수 열과 빈 값
- 종목코드·이벤트 유형·방향·출처 유형
- 날짜 형식과 `announce_date < effective_date`
- 같은 종목·유형·방향·적용일의 중복

파이프라인에서 사용하려면 `configs/pipeline.yaml`의 `events_path`를 설정한다.
구성종목 변화에서 만든 편입·편출 이벤트와 CSV를 대조해 공지 누락과 대응하지 않는 공지의 예시를 출력한다.
공지가 없는 이벤트는 `events_missing`에 따라 처리한다: `error`면 중단하고, `drop`(현재 설정)이면 빼고 진행하며 건수를 출력·`meta.json`·`hold_reasons`에 남긴다.
구성종목 변화가 없는 공지는 어느 쪽이든 중단한다.

## 공지 목록에서 만들기 (`from-notices`)

KRX 지수 공지 게시판의 공지 목록(공지 단위, 종목코드 없음)이 있으면, 구성종목 변화에서 만든 이벤트와 적용일로 짝지어
위 형식의 공지일 CSV를 만든다.

```bash
.venv/Scripts/python -m src.events from-notices --notices data/manual/events/<공지목록>.csv --source-type crawled
```

- 공지 목록 열: `변경적용일, 변경유형, 링크, 게시물번호, 공지일, 제목, 비고` (UTF-8, BOM 가능)
- 구성종목 이벤트는 `configs/pipeline.yaml`의 `legacy_dir`(또는 환경변수 `LEGACY_DATA_DIR`)와 `as_of`로 읽는다.
- 반영된 공지만 쓴다: 변경적용일이 있고, 변경유형에 괄호 표시(취소·미실시·일정변경)가 없고, 비고에 "미반영"이 없는 공지.
- 같은 적용일의 공지가 여럿이면 가장 이른 공지일을 쓴다 (시점 규칙상 보수적).
- `--source-type`: 공지 목록을 얻은 방식 (`crawled` 기본, 원문을 직접 확인해 작성했으면 `manual`). `collected_at`은 기본으로 공지 목록 파일의 수정 시각.
- 짝이 없는 이벤트는 CSV에 넣지 않고 적용일별로 출력한다. 파이프라인은 `events_missing`에 따라 이 이벤트를 빼거나 중단한다 (위 "검사" 절). 현재 상태는 CLAUDE.md 2.4.
- 결과 파일은 저장 직후 `validate`와 같은 형식 검사를 거친다.

## 공지 없는 이벤트 16건 (2026-11-19 확인, 개발용)

`from-notices`가 짝짓지 못해 파이프라인이 빼는 16건은 legacy 구성종목이 바뀐 **8일 × (편입 1 + 편출 1)** 이다 (`docs/DECISIONS.md` 2.7).

| 적용일 | 편출 종목 | 원인 |
|---|---|---|
| 2018-04-16 | 에프티이앤이 | 공지 목록에 공지 없음 (그 무렵 거래량 0) |
| 2019-04-10 | 케어젠 | 공지 목록에 공지 없음 (그 무렵 거래량 0) |
| 2020-04-22 | 제낙스 | 공지(2020-03-30)는 있으나 변경적용일 빈칸 — 조건부 반영 |
| 2022-03-16 | 오스템임플란트 | 공지(2022-02-18)는 있으나 변경적용일 빈칸 |
| 2023-04-06 | 국일제지 | 공지(2023-03-15)는 있으나 변경적용일 빈칸 |
| 2023-04-17 | 셀리버리 | 공지(2023-03-24)는 있으나 변경적용일 빈칸 |
| 2024-04-16 | 카나리아바이오(구 현대사료) | 공지(2024-03-22)는 있으나 변경적용일 빈칸 |
| 2024-05-02 | 엔케이맥스(구 에이티젠) | 공지(2024-04-08)는 있으나 변경적용일 빈칸 |

- 적용일은 legacy 구성종목 변화에서 찾은 날이고, 공지와의 대응은 종목명으로 맞췄다. **KRX 공지 원문으로 확인하기 전에는 CSV에 채우지 않는다** [미확인].
  원문으로 확인한 뒤 공지 목록의 `변경적용일`을 채우면 짝지어진다. `from-notices`의 `--source-type`은 파일 전체에 붙으므로,
  손으로 확인한 행은 `비고`에 원문 대조 사실을 남겨 구분한다 (행 단위 `manual` 표시는 현재 지원하지 않음).
- 공지 목록에는 반영되지 않은 공지도 있다: 2021-05-18 서부T&D(미실시), 2025-07 하나마이크론 회사분할(일정 변경 후 취소). 이들은 `from-notices`가 이미 제외한다.
