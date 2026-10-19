"""KRX Open API 날짜별 수집.

순서:
  1. start ~ as_of의 평일마다 KOSDAQ 시리즈 지수 일별시세(idx/kosdaq_dd_trd)
     → 지수가 비면 코스닥 일별매매정보도 비었는지 확인해야 휴장일로 본다. 휴장일은 0행 지수·종목 파일로 남겨
       다시 요청하지 않는다. 지수만 비고 종목 데이터가 있으면 실패로 기록한다.
  2. 지수 응답에 "코스닥 150" 행이 있는 날(= 거래일)마다 코스닥 일별매매정보(sto/ksq_bydd_trd)
  3. 매달 마지막 거래일마다 코스닥 종목기본정보(sto/ksq_isu_base_info) — 상장일(LIST_DD)용.
     과거 날짜로 조회하면 그날 상장돼 있던 종목 전체(나중에 상장폐지된 종목 포함)가 온다

응답 검증(실패하면 파일을 만들지 않고, 다시 실행하면 그 단위만 받는다): 필수 열, BAS_DD == 요청일,
종목코드 중복·빈 값, "코스닥 150" 행이 정확히 1개, 거래일인데 0행.
휴장일 이중 확인·응답 검증·validate는 이나영 topic/krx-openapi(src/crawl)의 방식을 가져왔다.

저장: data/raw/openapi/{kosdaq_index|kosdaq_daily|kosdaq_basic}/date=YYYYMMDD.parquet — 응답 원본 문자열 그대로.
Open API 가격은 수정주가가 아니라 그날 실제 가격이므로 받은 날짜에 따라 과거 값이 바뀌지 않는다.
"""
import pandas as pd
from tqdm import tqdm

from .client import ApiError, DailyLimit

ROOT = "raw/openapi"
SNAPSHOT = f"{ROOT}/snapshot.json"
INDEX_EP = "idx/kosdaq_dd_trd"
DAILY_EP = "sto/ksq_bydd_trd"
BASIC_EP = "sto/ksq_isu_base_info"
INDEX_NAME = "코스닥 150"
MAX_CONSECUTIVE_FAILURES = 10

INDEX_COLS = {"BAS_DD", "IDX_NM", "OPNPRC_IDX", "HGPRC_IDX", "LWPRC_IDX", "CLSPRC_IDX", "ACC_TRDVOL", "ACC_TRDVAL"}
DAILY_COLS = {"BAS_DD", "ISU_CD", "ISU_NM", "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "TDD_CLSPRC",
              "ACC_TRDVOL", "ACC_TRDVAL", "MKTCAP", "LIST_SHRS"}
BASIC_COLS = {"ISU_CD", "ISU_SRT_CD", "ISU_NM", "LIST_DD", "LIST_SHRS"}


def month_ends(trading):
    """거래일 목록(YYYYMMDD)에서 달마다 마지막 거래일."""
    last = {}
    for d in trading:
        last[d[:6]] = max(last.get(d[:6], d), d)
    return sorted(last.values())


def check_rows(rows, cols, bas_dd=None, unique=None, label=""):
    """응답 검증. 문제가 있으면 ValueError."""
    if not rows:
        raise ValueError(f"{label}: 거래일인데 빈 응답")
    missing = cols - set(rows[0])
    if missing:
        raise ValueError(f"{label}: 필수 열 누락 {sorted(missing)}")
    if bas_dd is not None:
        dates = {str(r.get("BAS_DD", "")) for r in rows}
        if dates != {bas_dd}:
            raise ValueError(f"{label}: 요청일 {bas_dd}, 응답일 {sorted(dates)[:3]}")
    if unique is not None:
        keys = [str(r.get(unique, "")).strip() for r in rows]
        if "" in keys or len(keys) != len(set(keys)):
            raise ValueError(f"{label}: {unique}가 비었거나 중복됨")


def check_index(rows, bas_dd):
    check_rows(rows, INDEX_COLS, bas_dd, label="지수")
    n = sum(str(r["IDX_NM"]).strip() == INDEX_NAME for r in rows)
    if n != 1:
        raise ValueError(f"지수: '{INDEX_NAME}' 행이 {n}개")


class Collector:
    def __init__(self, client, store, cfg, log=print):
        self.client = client
        self.store = store
        self.cfg = cfg
        self.log = log
        self.fetched = self.skipped = 0
        self.failed = []
        self._streak = 0

    def _guard(self, rel, fn):
        """fn()을 실행하고 실패하면 기록한 뒤 계속한다 (DailyLimit·ApiError·연속 실패는 멈춤)."""
        try:
            fn()
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
        self._streak = 0

    def _write(self, rel, rows):
        self.store.write(rel, rows)
        self.fetched += 1

    def index_day(self, d):
        rel = f"{ROOT}/kosdaq_index/date={d}.parquet"
        if self.store.exists(rel):
            self.skipped += 1
            return

        def fetch():
            rows = self.client.get(INDEX_EP, d)
            if rows:
                check_index(rows, d)
                self._write(rel, rows)
                return
            daily = self.client.get(DAILY_EP, d)  # 지수가 비면 종목도 비어야 휴장일
            if daily:
                raise ValueError("지수는 빈 응답인데 종목 데이터가 있음 — 휴장일로 볼 수 없음")
            self._write(f"{ROOT}/kosdaq_daily/date={d}.parquet", [])
            self._write(rel, [])
        self._guard(rel, fetch)

    def unit(self, rel, endpoint, bas_dd, check):
        rel = f"{ROOT}/{rel}"
        if self.store.exists(rel):
            self.skipped += 1
            return

        def fetch():
            rows = self.client.get(endpoint, bas_dd)
            check(rows)
            self._write(rel, rows)
        self._guard(rel, fetch)

    def trading_days(self, days):
        """저장된 지수 파일에 '코스닥 150' 행이 있는 날."""
        out = []
        for d in days:
            rel = f"{ROOT}/kosdaq_index/date={d}.parquet"
            if self.store.exists(rel):
                df = self.store.read(rel)
                if "IDX_NM" in df and (df["IDX_NM"].str.strip() == INDEX_NAME).any():
                    out.append(d)
        return out

    def run(self, limit=None, only_date=None):
        """종료 코드: 0 = 정상, 1 = 실패한 단위 있음, 2 = 하루 한도·요청 제한으로 멈춤 (다음 날 이어 받기)."""
        self.store.check_snapshot(SNAPSHOT, {"source": "KRX Open API", "as_of": self.cfg["as_of"],
                                             "start": self.cfg["start"]})
        days = [d.strftime("%Y%m%d") for d in pd.bdate_range(self.cfg["start"], self.cfg["as_of"])]
        if only_date is not None:
            d = pd.Timestamp(only_date).strftime("%Y%m%d")
            if d not in days:
                raise ValueError(f"--date {only_date}는 {self.cfg['start']} ~ {self.cfg['as_of']}의 평일이어야 합니다.")
            days = [d]
        if limit is not None:
            days = days[:limit]
        trading, stopped = [], False
        try:
            for d in tqdm(days, desc="kosdaq_index"):
                self.index_day(d)
            trading = self.trading_days(days)
            for d in tqdm(trading, desc="kosdaq_daily"):
                self.unit(f"kosdaq_daily/date={d}.parquet", DAILY_EP, d,
                          lambda rows, d=d: check_rows(rows, DAILY_COLS, d, "ISU_CD", "코스닥 일별매매"))
            for d in tqdm(month_ends(trading), desc="kosdaq_basic"):
                self.unit(f"kosdaq_basic/date={d}.parquet", BASIC_EP, d,
                          lambda rows: check_rows(rows, BASIC_COLS, None, "ISU_SRT_CD", "코스닥 종목기본"))
        except DailyLimit as e:
            self.log(str(e))
            stopped = True
        finally:
            n = self.store.write_manifest(self.cfg["as_of"])
            self.log(f"새로 받음 {self.fetched}, 건너뜀 {self.skipped}, 실패 {len(self.failed)}, "
                     f"API 호출 {self.client.calls} (오늘 누적 {self.client.budget.used()}), MANIFEST {n}행")
        if not stopped:
            holidays = sorted(set(days) - set(trading))
            by_year = pd.Series([d[:4] for d in holidays]).value_counts().sort_index().to_dict() if holidays else {}
            self.log(f"평일 {len(days)}, 거래일 {len(trading)}, 평일 휴장 {len(holidays)} (연도별 {by_year}) "
                     "— KRX 휴장일 수와 맞는지 확인")
        return 2 if stopped else (1 if self.failed else 0)


def validate_files(store, cfg, complete=False):
    """저장된 파일을 검사해 문제 목록을 돌려준다 (네트워크 없음).
    - 지수: 0행(휴장)이면 같은 날 0행 종목 파일이 있어야 함 / 아니면 '코스닥 150' 1행, BAS_DD == 날짜
    - 거래일 종목: 파일 존재, 0행 아님, ISU_CD 중복 없음, BAS_DD == 날짜, 거래가 있던 종목의 OHLC 관계
      (저가 ≤ 시가·종가 ≤ 고가). 거래정지 행(시가 0·거래량 0)은 제외
    - 종목기본: 거래일이 있는 달마다 파일, ISU_SRT_CD 중복 없음
    - complete=True: start ~ as_of의 모든 평일에 지수 파일이 있는지
    """
    issues = []
    root = store.path(ROOT)
    files = sorted((root / "kosdaq_index").glob("date=*.parquet"))
    if not files:
        return [f"{root / 'kosdaq_index'}: 수집 파일이 없습니다."]
    have = {p.stem[5:] for p in files}
    if complete:
        weekdays = [d.strftime("%Y%m%d") for d in pd.bdate_range(cfg["start"], cfg["as_of"])]
        missing = [d for d in weekdays if d not in have]
        if missing:
            issues.append(f"평일 지수 파일 {len(missing)}개 누락 (처음: {', '.join(missing[:5])})")
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    trading = []
    for p in files:
        d = p.stem[5:]
        idx = pd.read_parquet(p)
        daily_p = root / "kosdaq_daily" / f"date={d}.parquet"
        if idx.empty:
            if not daily_p.exists() or not pd.read_parquet(daily_p).empty:
                issues.append(f"{d}: 휴장일 표시(0행 지수)인데 0행 종목 파일이 없음")
            continue
        try:
            check_index(idx.to_dict("records"), d)
        except ValueError as e:
            issues.append(f"{d}: {e}")
            continue
        trading.append(d)
        if not daily_p.exists():
            issues.append(f"{d}: 거래일 종목 파일 없음")
            continue
        daily = pd.read_parquet(daily_p)
        try:
            check_rows(daily.to_dict("records"), DAILY_COLS, d, "ISU_CD", "코스닥 일별매매")
        except ValueError as e:
            issues.append(f"{d}: {e}")
            continue
        o, h, l, c = (num(daily[k]) for k in ("TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "TDD_CLSPRC"))
        active = (num(daily["ACC_TRDVOL"]) > 0) & (o > 0)
        bad = active & ((l > o) | (l > c) | (o > h) | (c > h) | (l > h))
        if bad.any():
            issues.append(f"{d}: OHLC 관계 오류 {int(bad.sum())}행 (예: {daily.loc[bad, 'ISU_CD'].head(3).tolist()})")
    for d in month_ends(trading):
        bp = root / "kosdaq_basic" / f"date={d}.parquet"
        if not bp.exists():
            issues.append(f"{d}: 월말 종목기본 파일 없음")
            continue
        try:
            check_rows(pd.read_parquet(bp).to_dict("records"), BASIC_COLS, None, "ISU_SRT_CD", "코스닥 종목기본")
        except ValueError as e:
            issues.append(f"{d}: {e}")
    return issues
