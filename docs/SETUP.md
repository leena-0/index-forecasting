# 실행 환경 설정 (처음 한 번)

이 레포의 수집기(`src/krxapi`)와 파이프라인(`src/pipeline`)을 돌리기 위한 준비 과정이다.
데이터 출처는 **KRX Open API**(공식 경로)다. `data.krx.co.kr` 화면 자동 수집은 약관 위반이라 쓰지 않는다 (`CLAUDE.md` 9.2).

## 1. Python 가상환경

Python **3.11** 기준 (`.python-version`).

```bash
# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

설치 확인: `python -m unittest` → 모든 테스트 통과 (네트워크·키 없이 돈다).

## 2. KRX Open API 인증키 받기

1. [openapi.krx.co.kr](https://openapi.krx.co.kr)에 로그인한다 (회원가입 필요).
   KRX Data Marketplace 계정과 같은 계정인지는 [미확인].
2. **마이페이지**에서 **인증키**를 발급받는다 (40자). 공개 자료 기준으로 유효기간 1년, 오래 쓰지 않으면 삭제될 수 있다 [미확인].
3. **서비스 목록**에서 아래 세 API를 각각 열어 **"API 이용신청"**을 누른다. 인증키 발급과 이용신청은 **별개**다.

   | API | 경로 | 용도 |
   |---|---|---|
   | 주식 → 코스닥 일별매매정보 | `sto/ksq_bydd_trd` | 종목별 시가·고가·저가·종가, 거래량, 거래대금, 시가총액, 상장주식수 |
   | 지수 → KOSDAQ 시리즈 일별시세정보 | `idx/kosdaq_dd_trd` | 코스닥 150 지수 일별 |
   | 주식 → 코스닥 종목기본정보 | `sto/ksq_isu_base_info` | 상장일 |

4. 마이페이지에서 세 API가 **승인** 상태가 될 때까지 기다린다 (몇 시간 ~ 1일). 승인 전에 부르면 401 `Unauthorized API Call`이 난다.

이용 조건 (요약, 원문 확인 필요): **키당 하루 10,000건**, 비상업적 이용, **받은 데이터를 제3자에게 제공 금지**, 결과물에 출처
"한국거래소 통계정보" 표시. → **팀원마다 자기 키로 직접 수집**하고, 키와 데이터를 서로 넘기지 않는다.

## 3. `.env` 만들기

레포 루트에서 `.env.example`을 `.env`로 복사한 뒤 값을 채운다. **`.env`는 절대 커밋하지 않는다** (`.gitignore`에 있음).

```bash
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

| 변수 | 필수 | 설명 | 예 |
|---|---|---|---|
| `KRX_OPENAPI_KEY` | 수집할 때 | 2단계에서 받은 인증키. 요청 헤더 `AUTH_KEY`로만 쓰인다 | (40자) |
| `COLLECTOR_ID` | 수집할 때 | 수집자 구분. 사람마다 다르게 — `MANIFEST.csv`·진행 기록에 남는다 | `ymjeon`, `nylee` |
| `LEGACY_DATA_DIR` | 파이프라인 `prepare`할 때 | 이전 개인 수집 데이터 폴더(`kosdaq150_forecast/data`). 구성종목·이벤트를 아직 여기서 읽는다 (개발용) | `C:/.../kosdaq150_forecast/data` |

- 값에 따옴표나 `=` 양옆 공백을 넣지 않는다. 한 줄에 하나.
- 이미 터미널 환경변수로 같은 이름이 설정돼 있으면 그 값이 우선한다 (`.env`가 덮어쓰지 않음).

## 4. 처음 실행

```bash
# 1) 하루만 받아 보기 (API 3~4건) → 2) 파일 검사
.venv/Scripts/python -m src.krxapi collect --date 2015-07-13
.venv/Scripts/python -m src.krxapi validate

# 3) 전체 수집 (약 5,900건, 끊기면 같은 명령으로 이어 받음) → 4) 전체 검사 → 5) MANIFEST 해시 대조
.venv/Scripts/python -m src.krxapi collect
.venv/Scripts/python -m src.krxapi validate --complete
.venv/Scripts/python -m src.krxapi verify
```

(macOS/Linux는 `.venv/bin/python`.) 수집 설정은 `configs/krxapi.yaml`, 결과는 `data/`(git 제외)에 저장된다.
자세한 동작은 `docs/PIPELINE.md` 8절.

ETF PDF 월말 원본 검증·표준화는 `docs/ETF_PDF.md`를 따른다. CSV를 XLSX로 변환하지 않고 두 형식을 그대로 읽는다.

**기준일이 다른 데이터를 섞지 않는다**: `data/raw/openapi/snapshot.json`의 `as_of`·`start`·`source`가 config와 다르면
수집을 거부한다. 처음부터 다시 받으려면 `data/raw/openapi`, `data/_state`, `data/MANIFEST.csv`를 지우고 실행한다.

## 5. 자주 나는 오류

| 메시지 | 원인 | 해결 |
|---|---|---|
| `401 Unauthorized Key` | 인증키가 없거나 틀림 | `.env`의 `KRX_OPENAPI_KEY` 확인 |
| `401 Unauthorized API Call` | 그 API의 이용신청이 승인되지 않음 | 2단계 3·4 — 서비스 목록에서 이용신청, 승인 확인 |
| `.env에 COLLECTOR_ID를 적어 주세요` | `COLLECTOR_ID` 없음 | `.env`에 추가. 넣었는데도 나오면 터미널에 빈 환경변수가 있는지 확인 (새 터미널에서 실행) |
| `오늘 호출 한도 …에 닿았습니다` / `429` (종료 코드 2) | 하루 한도(`daily_limit`, 기본 9,500) 또는 서버 요청 제한 | 다음 날 같은 명령을 다시 실행하면 이어 받는다. 오늘 호출 수: `data/_state/openapi-requests-날짜.json` |
| `스냅샷 설정과 config가 다릅니다` | 기준일 등이 다른 기존 수집 데이터가 있음 | 4절 마지막 문단 |
| `configs/pipeline.yaml의 legacy_dir 또는 환경변수 LEGACY_DATA_DIR를 설정하세요` | 파이프라인용 기존 데이터 경로 없음 | `.env`에 `LEGACY_DATA_DIR` 추가 |

## 6. 커밋하면 안 되는 것

`.env`, `data/` 아래 파일(`data/README.md`, `data/MANIFEST.csv` 제외), `.venv/`. 인증키를 코드·문서·채팅에 붙이지 않는다.
