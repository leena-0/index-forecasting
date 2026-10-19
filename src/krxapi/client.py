"""KRX Open API (openapi.krx.co.kr) 클라이언트.

- 호출: GET https://data-dbg.krx.co.kr/svc/apis/{endpoint}?basDd=YYYYMMDD, 헤더 AUTH_KEY=인증키
- 응답: {"OutBlock_1": [ {...}, ... ]} — 값은 모두 문자열
- 401: "Unauthorized Key"(키 없음·틀림), "Unauthorized API Call"(그 API 이용신청이 승인되지 않음) — 재시도하지 않는다
- 이용 조건: 키당 하루 10,000건. max_calls에 닿으면 DailyLimit으로 멈춘다 (다음 날 다시 실행하면 이어 받음)
"""
import time

import requests

BASE = "https://data-dbg.krx.co.kr/svc/apis/"


class ApiError(RuntimeError):
    pass


class DailyLimit(RuntimeError):
    pass


class OpenApiClient:
    def __init__(self, key, sleep_sec=0.5, retries=3, max_calls=9500, session_factory=requests.Session):
        self._key = key
        self.sleep_sec = sleep_sec
        self.retries = retries
        self.max_calls = max_calls
        self.calls = 0
        self.session = session_factory()

    def get(self, endpoint, bas_dd):
        """endpoint(예: 'sto/ksq_bydd_trd')의 bas_dd 하루치 OutBlock_1 목록."""
        last_err = None
        for attempt in range(self.retries):
            if self.calls >= self.max_calls:
                raise DailyLimit(f"이번 실행 호출 한도 {self.max_calls}건에 닿았습니다. 내일 다시 실행하세요.")
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
            if resp.status_code >= 500:
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "OutBlock_1" not in data:
                raise KeyError(f"{endpoint} 응답에 OutBlock_1이 없습니다. 받은 키: {sorted(data)}")
            return data["OutBlock_1"]
        raise RuntimeError(f"{endpoint} {bas_dd} 요청 {self.retries}회 실패: {last_err}")
