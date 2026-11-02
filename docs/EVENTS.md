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
.venv/bin/python -m src.events validate
```

다음을 검사한다.

- 필수 열과 빈 값
- 종목코드·이벤트 유형·방향·출처 유형
- 날짜 형식과 `announce_date < effective_date`
- 같은 종목·유형·방향·적용일의 중복

파이프라인에서 사용하려면 `configs/pipeline.yaml`의 `events_path`를 설정한다.
구성종목 변화에서 만든 모든 편입·편출 이벤트와 CSV가 일대일로 대응하지 않으면 파이프라인을 중단하고
공지 누락과 대응하지 않는 공지의 예시를 출력한다.
