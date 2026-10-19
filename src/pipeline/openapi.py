"""KRX Open API 수집 데이터(data/raw/openapi) → 공통 형식 (data_source = openapi).

가격·거래량·거래대금·시가총액·상장주식수와 지수는 Open API(공식 경로)에서, 구성종목·이벤트는
Open API가 제공하지 않으므로 당분간 기존 개발용 데이터(legacy)에서 가져온다 → 결과는 여전히 개발용.

공통 형식은 legacy.load()와 같고 다음이 추가된다.
  prices.value         거래대금
  prices.corp_action   분할·증자 등으로 가격 기준이 바뀐 것으로 보이는 날 (True)
                       Open API 가격은 수정주가가 아니므로, 이날의 야간 수익률 ln(O_t / C_{t-1})은
                       실제 수익률이 아니다 → target.forward_yz가 이날이 들어간 타깃 구간을 결측으로 둔다.
                       판정: |ln(O_t / C_{t-1})| > ln(1 + gap_limit)  (가격제한폭 ±30% 밖 = 기준가 조정)
                             또는 상장주식수 비율 ≥ shares_ratio 또는 ≤ 1/shares_ratio
  early_end            universe(구성종목 이력에 나온 종목) 중 코스닥 데이터가 as_of 전에 끝나는 종목
                       kind = moved(기존 데이터는 계속됨 → 유가증권 이전 상장 추정) / ended(같이 끝남 → 상장폐지 추정)
  index                코스닥 150 지수 일별 (date, open, high, low, close, volume, value, mcap)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import legacy

INDEX_NAME = "코스닥 150"
COLS = {"TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low", "TDD_CLSPRC": "close",
        "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value", "MKTCAP": "mcap", "LIST_SHRS": "shares"}
IDX_COLS = {"OPNPRC_IDX": "open", "HGPRC_IDX": "high", "LWPRC_IDX": "low", "CLSPRC_IDX": "close",
            "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value", "MKTCAP": "mcap"}
PRICE_COLS = ["open", "high", "low", "close", "volume"]
MOVED_GAP_DAYS = 7  # 기존 데이터가 코스닥 데이터보다 이만큼(일) 넘게 이어지면 이전 상장으로 본다


def num(s):
    """KRX 문자열 → 숫자. 쉼표 제거, '', '-'는 결측."""
    return pd.to_numeric(s.astype(str).str.replace(",", "").replace({"": None, "-": None}), errors="coerce")


def _read_dates(folder, as_of):
    files = sorted(Path(folder).glob("date=*.parquet"))
    return [(pd.Timestamp(p.stem[5:]), p) for p in files if pd.Timestamp(p.stem[5:]) <= as_of]


def flag_corp_actions(prices, gap_limit=0.30, shares_ratio=1.5):
    """종목별로 직전 행(직전 거래일)과 비교해 corp_action 열을 붙인다. prices는 ticker, date로 정렬돼 있어야 한다."""
    g = prices.groupby("ticker", sort=False)
    prev_close, prev_shares = g["close"].shift(1), g["shares"].shift(1)
    gap = np.log(prices["open"] / prev_close).abs() > np.log(1 + gap_limit)
    ratio = prices["shares"] / prev_shares
    jump = (ratio >= shares_ratio) | (ratio <= 1 / shares_ratio)
    return prices.assign(corp_action=(gap | jump).fillna(False).astype(bool))


def load(openapi_dir, legacy_dir, as_of, placeholder_announce="effective", gap_limit=0.30, shares_ratio=1.5):
    d = Path(openapi_dir)
    as_of = pd.Timestamp(as_of)

    idx_rows = []
    for date, p in _read_dates(d / "kosdaq_index", as_of):
        df = pd.read_parquet(p)
        if "IDX_NM" in df:
            idx_rows.append(df[df["IDX_NM"] == INDEX_NAME].assign(date=date))
    index = pd.concat(idx_rows, ignore_index=True)
    index = index.rename(columns=IDX_COLS)[["date"] + list(IDX_COLS.values())]
    for c in IDX_COLS.values():
        index[c] = num(index[c])
    calendar = pd.DatetimeIndex(index["date"]).sort_values()

    parts = [pd.read_parquet(p).assign(date=date) for date, p in _read_dates(d / "kosdaq_daily", as_of)]
    raw = pd.concat(parts, ignore_index=True)
    missing_days = sorted(set(calendar) - set(raw["date"]))
    if missing_days:
        raise ValueError(f"지수는 있는데 종목 데이터가 없는 거래일 {len(missing_days)}일: {missing_days[:5]} … 다시 수집하세요.")
    prices = raw.rename(columns={"ISU_CD": "ticker"})[["date", "ticker"] + list(COLS)].rename(columns=COLS)
    for c in COLS.values():
        prices[c] = num(prices[c])
    prices = prices.sort_values(["ticker", "date"], ignore_index=True)
    # 거래정지: 시가·고가·저가가 0·결측이거나 거래량 0 → OHLC·거래량 결측 (legacy와 같은 규칙)
    halt = (prices[["open", "high", "low"]].fillna(0) == 0).any(axis=1) | (prices["volume"].fillna(0) == 0)
    prices.loc[halt, PRICE_COLS] = np.nan
    prices = flag_corp_actions(prices, gap_limit, shares_ratio)

    old = legacy.load(legacy_dir, as_of, placeholder_announce)
    members, events = old["members"], old["events"]
    last_member_day = members["date"].max()

    universe = sorted(set(members["ticker"]))
    k_last = prices.groupby("ticker")["date"].max()
    l_last = old["prices"].groupby("ticker")["date"].max()
    end = min(calendar[-1], last_member_day)
    rows = []
    for t in universe:
        kl = k_last.get(t)
        if kl is None or kl < end:
            ll = l_last.get(t)
            moved = ll is not None and (kl is None or ll > kl + pd.Timedelta(days=MOVED_GAP_DAYS))
            kind = "moved" if moved else "ended"
            rows.append({"ticker": t, "kosdaq_last": kl, "legacy_last": ll, "kind": kind})
    early_end = pd.DataFrame(rows, columns=["ticker", "kosdaq_last", "legacy_last", "kind"])

    return {"calendar": calendar, "prices": prices, "members": members, "events": events,
            "index": index, "early_end": early_end, "members_until": last_member_day}
