"""기존 개인 수집 데이터(kosdaq150_forecast/data) → 공통 형식. 개발용 (data_source = dev_legacy).

이 데이터는 pykrx(KRX 화면)·네이버 차트를 자동 수집한 것이라 결과 수치로 보고하지 않는다.
공식 경로 데이터가 생기면 같은 형식을 돌려주는 어댑터를 새로 만들고 이 파일은 그대로 둔다.

읽는 파일 (모두 읽기만 함):
  raw/index_2203.csv                    지수 일별 → 거래일 달력
  raw/aie_history/stocks/*.csv          종목별 OHLCV (네이버 수정주가)  ┐ 합쳐서 386종목
  raw/stocks/*.csv                      종목별 OHLCV (같은 형식)        ┘
  raw/aie_history/mcap_daily.csv        종목별 일별 시가총액·상장주식수 (KRX)
  raw/aie_history/snapshots_2203.csv    거래일별 구성종목
  processed/events.csv                  편입·편출 이벤트 (공지일 없음)

공통 형식 (load()가 돌려주는 dict):
  calendar   DatetimeIndex — 거래일 (as_of까지)
  prices     DataFrame[date, ticker, open, high, low, close, volume, mcap, shares]
             거래정지일(시가·고가·저가 0, 또는 거래량 0)은 open·high·low·close·volume을 NaN으로 둔다 (보간 없음)
  members    DataFrame[date, ticker] — 그날의 구성종목
  events     DataFrame[event_id, ticker, event_type, direction, announce_date, effective_date, announce_source]
             event_type: 정기변경 / 수시변경 / 특례편입, direction: 편입 / 편출
"""
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_TYPES = {  # 기존 events.csv 표기 → (event_type, direction). initial(지수 산출 시작 시 구성)은 이벤트가 아님
    "regular_add": ("정기변경", "편입"), "regular_del": ("정기변경", "편출"),
    "other_add": ("수시변경", "편입"), "other_del": ("수시변경", "편출"),
    "special_add": ("특례편입", "편입"),
}
PRICE_COLS = ["open", "high", "low", "close", "volume"]


def load(legacy_dir, as_of, placeholder_announce="effective"):
    d = Path(legacy_dir)
    as_of = pd.Timestamp(as_of)

    idx = pd.read_csv(d / "raw/index_2203.csv", parse_dates=["date"])
    calendar = pd.DatetimeIndex(idx["date"][idx["date"] <= as_of]).sort_values()

    files = sorted((d / "raw/aie_history/stocks").glob("*.csv")) + sorted((d / "raw/stocks").glob("*.csv"))
    parts = []
    for p in files:
        s = pd.read_csv(p, parse_dates=["date"], usecols=["date"] + PRICE_COLS)
        s.insert(1, "ticker", p.stem)
        parts.append(s)
    prices = pd.concat(parts, ignore_index=True)
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("같은 (날짜, 종목)이 두 파일에 있습니다.")
    # 거래정지: 시가·고가·저가가 0이거나, OHLC가 직전 종가로 채워지고 거래량만 0인 날 (거래가 없으면 가격 정보도 없음)
    halt = (prices[["open", "high", "low"]] == 0).any(axis=1) | (prices["volume"] == 0)
    prices[PRICE_COLS] = prices[PRICE_COLS].astype(float)
    prices.loc[halt, PRICE_COLS] = np.nan

    mcap = pd.read_csv(d / "raw/aie_history/mcap_daily.csv", dtype={"date": str, "ticker": str})
    mcap["date"] = pd.to_datetime(mcap["date"], format="%Y%m%d")
    prices = prices.merge(mcap, on=["date", "ticker"], how="left")
    prices = prices[prices["date"] <= as_of].sort_values(["ticker", "date"], ignore_index=True)

    snap = pd.read_csv(d / "raw/aie_history/snapshots_2203.csv", dtype=str)
    members = pd.DataFrame({"date": pd.to_datetime(snap["date"], format="%Y%m%d"), "ticker": snap["ticker"]})
    members = members[members["date"] <= as_of].reset_index(drop=True)

    ev = pd.read_csv(d / "processed/events.csv", dtype=str)
    ev = ev[ev["event_type"] != "initial"]
    unknown = set(ev["event_type"]) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f"알 수 없는 event_type: {unknown}")
    events = pd.DataFrame({
        "ticker": ev["ticker"].values,
        "event_type": [EVENT_TYPES[t][0] for t in ev["event_type"]],
        "direction": [EVENT_TYPES[t][1] for t in ev["event_type"]],
        "effective_date": pd.to_datetime(ev["effective_date"]).values,
    })
    if placeholder_announce != "effective":
        raise ValueError("placeholder_announce는 지금 'effective'만 지원합니다.")
    events["announce_date"] = events["effective_date"]  # 임시 공지일 (개발용)
    events["announce_source"] = "placeholder"
    events = events[events["effective_date"] <= as_of].sort_values(["effective_date", "ticker"], ignore_index=True)
    events.insert(0, "event_id", range(len(events)))
    events = events[["event_id", "ticker", "event_type", "direction", "announce_date", "effective_date",
                     "announce_source"]]
    return {"calendar": calendar, "prices": prices, "members": members, "events": events}


def check_events(events, members, calendar):
    """편입 이벤트는 편입일에 구성종목이고 직전 거래일에는 아니어야 한다 (편출은 반대).
    맞지 않는 이벤트 목록을 돌려준다 (CLAUDE.md 2.4 결합 검증의 구성종목 쪽)."""
    S = members.groupby("date")["ticker"].apply(set)
    bad = []
    for e in events.itertuples():
        pos = calendar.get_indexer([e.effective_date])[0]
        if pos <= 0:
            bad.append((e.event_id, "편입일이 달력에 없거나 첫날"))
            continue
        now, prev = S.get(e.effective_date, set()), S.get(calendar[pos - 1], set())
        ok = (e.ticker in now and e.ticker not in prev) if e.direction == "편입" else \
             (e.ticker not in now and e.ticker in prev)
        if not ok:
            bad.append((e.event_id, "구성종목 변화와 맞지 않음"))
    return bad
