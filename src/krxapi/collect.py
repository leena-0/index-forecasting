"""KRX Open API 날짜별 수집.

순서:
  1. start ~ as_of의 평일마다 KOSDAQ 시리즈 지수 일별시세(idx/kosdaq_dd_trd)
     → 휴장일은 빈 응답(0행 파일로 저장해 다시 요청하지 않음)
  2. 지수 응답에 "코스닥 150" 행이 있는 날(= 거래일)마다 코스닥 일별매매정보(sto/ksq_bydd_trd)
     → 거래일인데 빈 응답이면 실패로 기록하고 파일을 만들지 않는다 (다시 실행하면 그날만 받음)

저장: data/raw/openapi/{kosdaq_index|kosdaq_daily}/date=YYYYMMDD.parquet — 응답 원본 문자열 그대로.
Open API 가격은 수정주가가 아니라 그날 실제 가격이므로 받은 날짜에 따라 과거 값이 바뀌지 않는다.
"""
import pandas as pd
from tqdm import tqdm

from .client import ApiError, DailyLimit

ROOT = "raw/openapi"
SNAPSHOT = f"{ROOT}/snapshot.json"
INDEX_EP = "idx/kosdaq_dd_trd"
DAILY_EP = "sto/ksq_bydd_trd"
INDEX_NAME = "코스닥 150"
MAX_CONSECUTIVE_FAILURES = 10


class Collector:
    def __init__(self, client, store, cfg, log=print):
        self.client = client
        self.store = store
        self.cfg = cfg
        self.log = log
        self.fetched = self.skipped = 0
        self.failed = []
        self._streak = 0

    def unit(self, rel, endpoint, bas_dd, require_rows):
        rel = f"{ROOT}/{rel}"
        if self.store.exists(rel):
            self.skipped += 1
            return
        try:
            rows = self.client.get(endpoint, bas_dd)
            if require_rows and not rows:
                raise ValueError("거래일인데 빈 응답")
            self.store.write(rel, rows)
        except Exception as e:  # noqa: BLE001 — 단위 하나의 실패로 전체를 멈추지 않는다
            self.store.log_failure(rel, e)
            if isinstance(e, (DailyLimit, ApiError)):  # 계속해도 소용없음 → 멈춘다
                raise
            self.failed.append(rel)
            self.log(f"  실패 {rel}: {type(e).__name__}: {e}")
            self._streak += 1
            if self._streak >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f"연속 {self._streak}회 실패 — 상태를 확인한 뒤 다시 실행하세요.") from e
            return
        self.fetched += 1
        self._streak = 0

    def trading_days(self, days):
        """저장된 지수 파일에 '코스닥 150' 행이 있는 날."""
        out = []
        for d in days:
            rel = f"{ROOT}/kosdaq_index/date={d}.parquet"
            if self.store.exists(rel):
                df = self.store.read(rel)
                if "IDX_NM" in df and (df["IDX_NM"] == INDEX_NAME).any():
                    out.append(d)
        return out

    def run(self, limit=None):
        self.store.check_snapshot(SNAPSHOT, {"as_of": self.cfg["as_of"], "start": self.cfg["start"]})
        days = [d.strftime("%Y%m%d") for d in pd.bdate_range(self.cfg["start"], self.cfg["as_of"])]
        if limit is not None:
            days = days[:limit]
        try:
            for d in tqdm(days, desc="kosdaq_index"):
                self.unit(f"kosdaq_index/date={d}.parquet", INDEX_EP, d, require_rows=False)
            trading = self.trading_days(days)
            for d in tqdm(trading, desc="kosdaq_daily"):
                self.unit(f"kosdaq_daily/date={d}.parquet", DAILY_EP, d, require_rows=True)
        finally:
            n = self.store.write_manifest(self.cfg["as_of"])
            self.log(f"새로 받음 {self.fetched}, 건너뜀 {self.skipped}, 실패 {len(self.failed)}, "
                     f"API 호출 {self.client.calls}, MANIFEST {n}행")
        holidays = sorted(set(days) - set(trading))
        by_year = pd.Series([d[:4] for d in holidays]).value_counts().sort_index().to_dict() if holidays else {}
        self.log(f"평일 {len(days)}, 거래일 {len(trading)}, 평일 휴장 {len(holidays)} (연도별 {by_year}) "
                 "— KRX 휴장일 수와 맞는지 확인")
        return trading
