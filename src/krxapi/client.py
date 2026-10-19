"""KRX Open API (openapi.krx.co.kr) 클라이언트.

- 호출: GET https://data-dbg.krx.co.kr/svc/apis/{endpoint}?basDd=YYYYMMDD, 헤더 AUTH_KEY=인증키
- 응답: {"OutBlock_1": [ {...}, ... ]} — 값은 모두 문자열
- 401: "Unauthorized Key"(키 없음·틀림), "Unauthorized API Call"(그 API 이용신청이 승인되지 않음) — 재시도하지 않는다
- 429: 서버의 요청 제한 → 그날 수집을 멈춘다
- 이용 조건: 키당 하루 10,000건. RequestBudget이 날짜별 호출 수를 파일에 남겨, 같은 날 여러 번 실행해도 합산한다.
  (이나영 topic/krx-openapi의 RequestBudget 방식을 가져옴)
"""
import datetime as dt
import json
import time
from pathlib import Path

import requests

BASE = "https://data-dbg.krx.co.kr/svc/apis/"


class ApiError(RuntimeError):
    pass


class DailyLimit(RuntimeError):
    pass


class RequestBudget:
    """날짜별 호출 수를 state_dir/openapi-requests-YYYY-MM-DD.json에 기록한다 (로컬 전용, git 제외)."""

    def __init__(self, state_dir, limit=9500, today=dt.date.today):
        self.state_dir = Path(state_dir)
        self.limit = int(limit)
        self._today = today

    @property
    def path(self):
        return self.state_dir / f"openapi-requests-{self._today().isoformat()}.json"

    def used(self):
        if not self.path.exists():
            return 0
        return int(json.loads(self.path.read_text(encoding="utf-8"))["requests"])

    def consume(self):
        n = self.used()
        if n >= self.limit:
            raise DailyLimit(f"오늘 호출 한도 {self.limit:,}건에 닿았습니다. 내일 다시 실행하면 이어 받습니다.")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"date": self._today().isoformat(), "requests": n + 1}), encoding="utf-8")
        tmp.replace(self.path)


class OpenApiClient:
    def __init__(self, key, budget, sleep_sec=0.5, retries=3, session_factory=requests.Session):
        self._key = key
        self.budget = budget
        self.sleep_sec = sleep_sec
        self.retries = retries
        self.calls = 0  # 이번 실행에서 보낸 요청 수
        self.session = session_factory()

    def get(self, endpoint, bas_dd):
        """endpoint(예: 'sto/ksq_bydd_trd')의 bas_dd 하루치 OutBlock_1 목록."""
        last_err = None
        for attempt in range(self.retries):
            self.budget.consume()
            time.sleep(self.sleep_sec)
            self.calls += 1
            try:
                resp = self.session.get(BASE + endpoint, params={"basDd": bas_dd},
                                        headers={"AUTH_KEY": self._key}, timeout=30)
            except requests.RequestException as e:
                last_err = e
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 401:
                try:
                    msg = resp.json().get("respMsg", "")
                except ValueError:
                    msg = resp.text[:100]
                raise ApiError(f"{endpoint} 401 {msg} — 인증키와 API 이용신청 승인을 확인하세요.")
            if resp.status_code == 429:
                raise DailyLimit(f"{endpoint} 429 요청 제한 (Retry-After={resp.headers.get('Retry-After')!r}) "
                                 "— 오늘 수집을 멈춥니다.")
            if resp.status_code >= 500:
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError as e:  # JSON이 아님 → 재시도
                last_err = e
                time.sleep(2 ** attempt)
                continue
            if not isinstance(data.get("OutBlock_1"), list):
                raise KeyError(f"{endpoint} 응답에 OutBlock_1 목록이 없습니다. 받은 키: {sorted(data)}")
            return data["OutBlock_1"]
        raise RuntimeError(f"{endpoint} {bas_dd} 요청 {self.retries}회 실패: {last_err}")
