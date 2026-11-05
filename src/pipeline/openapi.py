"""KRX Open API 수집 데이터(data/raw/openapi) → 공통 형식 (data_source = openapi).

가격·거래량·거래대금·시가총액·상장주식수와 지수는 Open API(공식 경로)에서, 구성종목·이벤트는
Open API가 제공하지 않으므로 당분간 기존 개발용 데이터(legacy)에서 가져온다 → 결과는 여전히 개발용.

공통 형식은 legacy.load()와 같고 다음이 추가된다.
  prices.cmp_prevdd_prc KRX 전일 대비(CMPPREVDD_PRC)를 숫자로 변환한 값
  prices.fluc_rt        KRX 등락률(FLUC_RT)을 숫자로 변환한 값
  prices.base_price     당일 기준가 = close - cmp_prevdd_prc
  prices.base_price_valid
                       기준가 수익률 계산 가능 여부. 첫 관측일·신규상장일·거래정지일·거래정지 직후·기준가 오류·
                       FLUC_RT 검증 실패면 False
  prices.base_price_after_halt
                       거래정지 직후(직전 거래일에 거래가 없거나 행이 없는 날) 여부
  prices.share_lag     기준가 하락 뒤 아직 반영되지 않은 상장주식수 변화 (로그, 0이면 시차 없음). add_share_lag 참고
  prices.value         거래대금
  prices.corp_action   기존 1.3/0.7·주식 수 1.5배 규칙. 기준가 방식과 비교하는 검증용 플래그
  early_end            universe(구성종목 이력에 나온 종목) 중 코스닥 데이터가 as_of 전에 끝나는 종목
                       kind = moved(기존 데이터는 계속됨 → 유가증권 이전 상장 추정) / ended(같이 끝남 → 상장폐지 추정)
  index                코스닥 150 지수 일별 (date, open, high, low, close, volume, value, mcap)
  listing              종목별 상장일 (ticker, list_date) — 월말 종목기본정보(kosdaq_basic)에서
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import legacy

INDEX_NAME = "코스닥 150"
COLS = {"TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low", "TDD_CLSPRC": "close",
        "CMPPREVDD_PRC": "cmp_prevdd_prc", "FLUC_RT": "fluc_rt", "ACC_TRDVOL": "volume",
        "ACC_TRDVAL": "value", "MKTCAP": "mcap", "LIST_SHRS": "shares"}
IDX_COLS = {"OPNPRC_IDX": "open", "HGPRC_IDX": "high", "LWPRC_IDX": "low", "CLSPRC_IDX": "close",
            "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value", "MKTCAP": "mcap"}
PRICE_COLS = ["open", "high", "low", "close", "volume"]
MOVED_GAP_DAYS = 7  # 기존 데이터가 코스닥 데이터보다 이만큼(일) 넘게 이어지면 이전 상장으로 본다
FLUC_RT_ATOL = 0.011  # FLUC_RT 소수 둘째 자리 반올림 오차(최대 0.005%p)에 여유를 둔다
SHARE_LAG_MIN = 0.05  # 이 크기(로그)를 넘는 기준가 하락만 주식수 시차 후보로 본다
SHARE_LAG_MATCH = 0.10  # 기준가 변화와 주식수 변화의 합이 이보다 작으면 같은 날 반영된 것으로 본다
SHARE_LAG_SETTLE = 0.5  # 미반영분의 이 비율 이상 주식수가 늘면 반영된 것으로 본다
SHARE_LAG_MAX_DAYS = 120  # 이 거래일 수 안에 반영되지 않으면 보정을 멈춘다


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
    ratio = prices["open"] / prev_close
    gap = (ratio > 1 + gap_limit) | (ratio < 1 - gap_limit)  # 기본 ratio > 1.3 또는 ratio < 0.7
    ratio = prices["shares"] / prev_shares
    jump = (ratio >= shares_ratio) | (ratio <= 1 / shares_ratio)
    return prices.assign(corp_action=(gap | jump).fillna(False).astype(bool))


def add_base_prices(prices, listing=None, fluc_rt_atol=FLUC_RT_ATOL, calendar=None):
    """KRX 일별 기준가와 사용 가능 여부를 붙인다.

    KRX의 CMPPREVDD_PRC는 당일 종가와 당일 기준가의 차이이므로 기준가는
    ``close - cmp_prevdd_prc``로 복원한다. FLUC_RT는 소수 둘째 자리까지 제공되므로
    절대 오차 허용치 안에서 ``100 * cmp_prevdd_prc / base_price``와 대조한다.
    거래정지 직후는 정지 기간의 변화가 하루 값에 몰리므로 계산하지 않는다 (CLAUDE.md 5.3).
    calendar를 주면 달력상 직전 거래일에 행이 없는 날도 거래정지 직후로 본다.
    계산한 값과 원본에서 변환한 열은 덮어쓰지 않고 모두 남긴다.
    """
    out = prices.copy()
    out["base_price"] = out["close"] - out["cmp_prevdd_prc"]
    base_ok = out["base_price"].notna() & (out["base_price"] > 0)
    expected = 100 * out["cmp_prevdd_prc"] / out["base_price"]
    formula_ok = base_ok & out["fluc_rt"].notna() & np.isclose(
        out["fluc_rt"], expected, rtol=0, atol=fluc_rt_atol, equal_nan=False)

    first = out.groupby("ticker", sort=False).cumcount().eq(0)
    new_listing = pd.Series(False, index=out.index)
    if listing is not None and not listing.empty:
        list_dates = listing.drop_duplicates("ticker").set_index("ticker")["list_date"]
        new_listing = out["date"].eq(out["ticker"].map(list_dates))
    active = (out[["open", "high", "low", "close"]].notna().all(axis=1)
              & (out[["open", "high", "low", "close"]] > 0).all(axis=1)
              & out["volume"].notna() & (out["volume"] > 0))
    prev_active = active.astype(float).groupby(out["ticker"], sort=False).shift(1).eq(1.0)
    if calendar is not None:
        pos = pd.Series(pd.DatetimeIndex(calendar).get_indexer(out["date"]), index=out.index)
        prev_active &= pos.groupby(out["ticker"], sort=False).diff().eq(1)
    after_halt = active & ~first & ~prev_active

    out["base_price_formula_valid"] = formula_ok.astype(bool)
    out["base_price_after_halt"] = after_halt.astype(bool)
    out["base_price_valid"] = (base_ok & formula_ok & active & ~first & ~new_listing & ~after_halt).astype(bool)
    return out


def add_share_lag(prices):
    """권리락·분할일과 상장주식수 반영일의 시차를 share_lag(로그)로 기록한다 (이슈 #36).

    가격은 권리락일에 바뀌지만 LIST_SHRS와 MKTCAP은 신주 상장일에 바뀌는 경우가 많아 그 사이 시가총액이 낮게 찍힌다.
    거래일 t, 직전 거래일 s에 대해 lf = ln(B_t / C_s), ls = ln(shares_t / shares_s), p = t까지의 미반영분:
      - p > 0이고 ls ≥ SHARE_LAG_SETTLE · p이면 반영된 것으로 보고 p = 0
      - 마지막 후보일에서 SHARE_LAG_MAX_DAYS 거래일이 지나면 p = 0
      - lf < −SHARE_LAG_MIN이고 |ls + lf| ≥ SHARE_LAG_MATCH이면 p에 −lf를 더한다
    보정값: ln(시가총액) + p, ln(거래량 / 상장주식수) − p. t까지 알 수 있는 값만 쓴다 (3장).
    기준가가 오른 날(병합 등)은 주식수가 먼저 바뀌기도 해서 다루지 않는다. 원본 mcap·shares는 바꾸지 않는다.
    prices는 ticker, date로 정렬돼 있어야 한다.
    """
    close, base, shares = (prices[c].to_numpy(dtype=float) for c in ("close", "base_price", "shares"))
    active = prices["open"].gt(0).to_numpy()
    out = np.zeros(len(prices))
    for idx in prices.groupby("ticker", sort=False).indices.values():
        p, age, last_close, last_shares = 0.0, 0, np.nan, np.nan
        for i in idx:
            if active[i]:
                ls = np.log(shares[i] / last_shares) if shares[i] != last_shares and last_shares > 0 and shares[i] > 0 else 0.0
                lf = np.log(base[i] / last_close) if base[i] != last_close and last_close > 0 and base[i] > 0 else 0.0
                if p > 0:
                    age += 1
                    if age > SHARE_LAG_MAX_DAYS or ls >= SHARE_LAG_SETTLE * p:
                        p = 0.0
                if lf < -SHARE_LAG_MIN and abs(ls + lf) >= SHARE_LAG_MATCH:
                    p, age = p - lf, 0
                last_close, last_shares = close[i], shares[i]
            out[i] = p
    return prices.assign(share_lag=out)


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

    basic = [pd.read_parquet(p, columns=["ISU_SRT_CD", "LIST_DD"]) for _, p in _read_dates(d / "kosdaq_basic", as_of)]
    if basic:
        b = pd.concat(basic, ignore_index=True).drop_duplicates()
        b = b[b["LIST_DD"].astype(str).str.strip() != ""]
        listing = (b.assign(list_date=pd.to_datetime(b["LIST_DD"], format="%Y%m%d"))
                   .groupby("ISU_SRT_CD")["list_date"].min().rename_axis("ticker").reset_index())
    else:
        listing = pd.DataFrame(columns=["ticker", "list_date"])

    prices = add_share_lag(add_base_prices(prices, listing, calendar=calendar))

    return {"calendar": calendar, "prices": prices, "members": members, "events": events,
            "index": index, "early_end": early_end, "members_until": last_member_day, "listing": listing}
