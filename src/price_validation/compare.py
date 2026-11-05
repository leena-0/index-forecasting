"""KRX 기준가 수익률과 기존 네이버 수정 OHLC 수익률을 날짜·종목별로 비교한다.

외부 수정주가는 검증에만 사용하며 파이프라인 입력을 대체하지 않는다. 가격 수준은 조정 기준일에
따라 달라질 수 있으므로 비교 대상은 종가 수익률과 야간 수익률이다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.openapi import add_base_prices, flag_corp_actions, num

RAW_COLS = ["open", "high", "low", "close", "volume"]
KRX_COLS = ["ISU_CD", "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "TDD_CLSPRC",
            "CMPPREVDD_PRC", "FLUC_RT", "ACC_TRDVOL", "LIST_SHRS"]
RENAMED = {"ISU_CD": "ticker", "TDD_OPNPRC": "open", "TDD_HGPRC": "high",
           "TDD_LWPRC": "low", "TDD_CLSPRC": "close", "CMPPREVDD_PRC": "cmp_prevdd_prc",
           "FLUC_RT": "fluc_rt", "ACC_TRDVOL": "volume", "LIST_SHRS": "shares"}


def _date_from_path(path):
    return pd.Timestamp(path.stem.removeprefix("date="))


def _krx_files(openapi_dir, start=None, end=None):
    files = sorted((Path(openapi_dir) / "kosdaq_daily").glob("date=*.parquet"))
    dated = [(_date_from_path(p), p) for p in files]
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None
    selected = [(d, p) for d, p in dated if (start is None or d >= start) and (end is None or d <= end)]
    if start is not None:
        previous = [(d, p) for d, p in dated if d < start]
        if previous:
            selected.insert(0, previous[-1])  # 시작일 수익률·기준가 변경 판정을 위한 직전 거래일
    return selected


def load_krx(openapi_dir, start=None, end=None, tickers=None):
    """Open API 원본에서 기준가 수익률과 기준가 변경 여부를 만든다."""
    wanted = set(tickers) if tickers else None
    parts = []
    for date, path in _krx_files(openapi_dir, start, end):
        # 휴장일 표시 파일은 0행·0열 parquet이므로 열을 지정해 읽으면 ArrowInvalid가 난다.
        raw = pd.read_parquet(path)
        if raw.empty:
            continue
        missing = set(KRX_COLS) - set(raw.columns)
        if missing:
            raise ValueError(f"{path}: KRX 필수 열 누락 {sorted(missing)}")
        raw = raw[KRX_COLS]
        if wanted is not None:
            raw = raw[raw["ISU_CD"].astype(str).isin(wanted)]
        if not raw.empty:
            parts.append(raw.assign(date=date))
    if not parts:
        raise ValueError("비교 기간에 KRX 일별 데이터가 없습니다.")

    prices = pd.concat(parts, ignore_index=True).rename(columns=RENAMED)
    prices["ticker"] = prices["ticker"].astype(str).str.zfill(6)
    for column in set(RENAMED.values()) - {"ticker"}:
        prices[column] = num(prices[column])
    prices = prices.sort_values(["ticker", "date"], ignore_index=True)
    halt = ((prices[["open", "high", "low"]].fillna(0) == 0).any(axis=1)
            | (prices["volume"].fillna(0) == 0))
    prices.loc[halt, RAW_COLS] = np.nan
    prices = flag_corp_actions(prices)
    prices = add_base_prices(prices)

    previous_close = prices.groupby("ticker", sort=False)["close"].shift(1)
    comparable_base = prices["base_price"].notna() & previous_close.notna()
    prices["base_adjustment"] = (comparable_base
                                  & ~np.isclose(prices["base_price"], previous_close,
                                                rtol=0, atol=1e-12, equal_nan=False))
    reference = prices["base_price"].where(prices["base_price_valid"])
    prices["krx_ret"] = np.log(prices["close"] / reference)
    prices["krx_overnight"] = np.log(prices["open"] / reference)

    if start is not None:
        prices = prices[prices["date"] >= pd.Timestamp(start)]
    if end is not None:
        prices = prices[prices["date"] <= pd.Timestamp(end)]
    return prices[["ticker", "date", "krx_ret", "krx_overnight", "base_price",
                   "base_price_valid", "base_adjustment", "corp_action"]].reset_index(drop=True)


def load_adjusted(adjusted_dirs, start=None, end=None, tickers=None):
    """기존 네이버 수정 OHLC CSV 폴더를 읽어 수정주가 수익률을 만든다."""
    wanted = set(tickers) if tickers else None
    files = []
    for folder in adjusted_dirs:
        files.extend(sorted(Path(folder).glob("*.csv")))
    parts = []
    for path in files:
        ticker = path.stem.zfill(6)
        if wanted is not None and ticker not in wanted:
            continue
        frame = pd.read_csv(path, parse_dates=["date"], usecols=["date"] + RAW_COLS)
        frame.insert(1, "ticker", ticker)
        parts.append(frame)
    if not parts:
        raise ValueError("수정주가 CSV가 없습니다.")
    prices = pd.concat(parts, ignore_index=True)
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("수정주가에 중복된 (종목, 날짜) 행이 있습니다.")
    prices[RAW_COLS] = prices[RAW_COLS].apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_values(["ticker", "date"], ignore_index=True)
    halt = ((prices[["open", "high", "low"]].fillna(0) == 0).any(axis=1)
            | (prices["volume"].fillna(0) == 0))
    prices.loc[halt, RAW_COLS] = np.nan
    previous_close = prices.groupby("ticker", sort=False)["close"].shift(1)
    prices["adjusted_ret"] = np.log(prices["close"] / previous_close)
    prices["adjusted_overnight"] = np.log(prices["open"] / previous_close)
    if start is not None:
        prices = prices[prices["date"] >= pd.Timestamp(start)]
    if end is not None:
        prices = prices[prices["date"] <= pd.Timestamp(end)]
    return prices[["ticker", "date", "adjusted_ret", "adjusted_overnight"]].reset_index(drop=True)


def compare_returns(krx, adjusted, tolerance=1e-4):
    """공통 (종목, 날짜)의 두 수익률과 오차를 반환한다. tolerance는 로그수익률 단위다."""
    out = krx.merge(adjusted, on=["ticker", "date"], how="inner", validate="one_to_one")
    out["day_type"] = np.where(out["base_adjustment"], "base_adjustment", "regular")
    for metric in ("ret", "overnight"):
        out[f"{metric}_error"] = out[f"krx_{metric}"] - out[f"adjusted_{metric}"]
        out[f"{metric}_abs_error"] = out[f"{metric}_error"].abs()
    valid = out[["krx_ret", "krx_overnight", "adjusted_ret", "adjusted_overnight"]].notna().all(axis=1)
    out = out[valid].copy()
    out["mismatch"] = ((out["ret_abs_error"] > tolerance)
                       | (out["overnight_abs_error"] > tolerance))
    return out.sort_values(["date", "ticker"], ignore_index=True)


def summarize(comparison):
    """전체·일반일·기준가 조정일별 평균·중앙값·최대 절대오차."""
    rows = []
    groups = [("all", comparison), *(comparison.groupby("day_type", sort=True))]
    for day_type, group in groups:
        for metric in ("ret", "overnight"):
            error = group[f"{metric}_error"].dropna()
            absolute = error.abs()
            rows.append({"day_type": day_type, "metric": metric, "n": len(error),
                         "mean_abs_error": absolute.mean(), "median_abs_error": absolute.median(),
                         "max_abs_error": absolute.max(), "mean_error": error.mean()})
    return pd.DataFrame(rows)
