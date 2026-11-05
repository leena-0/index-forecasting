"""공통 특성 패널 (CLAUDE.md 6.2 모델 입력). t일 특성은 t일 장 마감까지 알 수 있는 값만 쓴다 (3장).

종목·거래일(t)마다
    o_t = ln(O_t / R_t),      c_t = ln(C_t / O_t),  u_t = ln(H_t / O_t),  d_t = ln(L_t / O_t)
    R_t = Open API는 당일 KRX 기준가, legacy는 C_{t−1}
    rv_d  = o_t² + u_t(u_t − c_t) + d_t(d_t − c_t)     일별 분산 대용치: 야간 수익률 제곱 + Rogers–Satchell
    rv_w  = (1/5)  Σ_{j=0}^{4}  rv_d_{t−j}             HAR 구성요소 (Corsi, 2009)
    rv_m  = (1/22) Σ_{j=0}^{21} rv_d_{t−j}
    yz_5, yz_22 = 구간 t−h+1 ~ t의 Yang–Zhang 분산 (타깃과 같은 정의 — origin t−h의 타깃, target.forward_yz)
    ret   = ln(C_t / R_t)                                기준가 대비 종가 수익률
    log_turnover = ln(volume_t / shares_t)
    log_mcap     = ln(mcap_t)
    is_member    = t일 구성종목 여부
    n_obs        = 종목의 첫 행부터 t까지 달력상 거래일 수 (이력 길이, CLAUDE.md 6.3)

결측은 보간하지 않는다: 창 안에 결측(거래정지, 상장 전)이 하나라도 있으면 NaN.
Open API는 base_price_valid가 False인 날의 o_t와 ret을 NaN으로 둔다.
legacy에서 corp_action 열이 있으면 기존 규칙대로 그날의 o_t와 ret을 NaN으로 둔다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .target import forward_yz

FEATURES = ["rv_d", "rv_w", "rv_m", "yz_5", "yz_22", "ret", "log_turnover", "log_mcap", "is_member", "n_obs"]


def build_features(prices, calendar, members, demean=True):
    """prices: DataFrame[date, ticker, open, high, low, close, volume, mcap, shares]
    Open API 입력은 base_price, base_price_valid를, legacy 호환 입력은 corp_action을 추가로 가질 수 있다.
    members: DataFrame[date, ticker]. 돌려주는 것: DataFrame[ticker, date] + FEATURES."""
    out = []
    for ticker, g in prices.groupby("ticker", sort=True):
        g = g.set_index("date").sort_index()
        days = calendar[(calendar >= g.index[0]) & (calendar <= g.index[-1])]
        g = g.reindex(days)
        if "base_price" in g:
            reference = g["base_price"]
            if "base_price_valid" in g:
                reference = reference.where(g["base_price_valid"].fillna(False).astype(bool))
        else:
            reference = g["close"].shift(1)
        o = np.log(g["open"] / reference)
        ret = np.log(g["close"] / reference)
        if "base_price" not in g and "corp_action" in g:
            ca = g["corp_action"].fillna(False).astype(bool)
            o, ret = o.mask(ca), ret.mask(ca)
        c = np.log(g["close"] / g["open"])
        u = np.log(g["high"] / g["open"])
        d = np.log(g["low"] / g["open"])
        rv_d = o ** 2 + u * (u - c) + d * (d - c)
        out.append(pd.DataFrame({
            "ticker": ticker, "date": days, "rv_d": rv_d.values,
            "rv_w": rv_d.rolling(5, min_periods=5).mean().values,
            "rv_m": rv_d.rolling(22, min_periods=22).mean().values,
            "ret": ret.values, "log_turnover": np.log(g["volume"] / g["shares"]).values,
            "log_mcap": np.log(g["mcap"]).values, "n_obs": np.arange(1, len(days) + 1)}))
    f = pd.concat(out, ignore_index=True)
    for h in (5, 22):  # 구간 t−h+1 ~ t = origin t−h의 타깃
        yz = forward_yz(prices, calendar, h, demean)
        yz["yz_past"] = yz.groupby("ticker")["yz_var"].shift(h)
        f = f.merge(yz[["ticker", "origin_date", "yz_past"]].rename(
            columns={"origin_date": "date", "yz_past": f"yz_{h}"}), on=["ticker", "date"], how="left")
    m = members[["date", "ticker"]].assign(is_member=True)
    f = f.merge(m, on=["date", "ticker"], how="left")
    f["is_member"] = f["is_member"].fillna(False).astype(bool)
    return f[["ticker", "date"] + FEATURES]


def load_panel(data_dir, h):
    """prepare 결과 폴더 → 모델 입력 패널 (특성 + 타깃 + 분할).
    열: ticker, origin_date, horizon, FEATURES, yz_var, log_var(= ln yz_var, yz_var ≤ 0이면 NaN), split, embargo.
    학습·평가에 쓸 수 있는 행 = log_var가 결측이 아니고 embargo == False."""
    d = Path(data_dir)
    f = pd.read_parquet(d / "features.parquet").rename(columns={"date": "origin_date"})
    t = pd.read_parquet(d / f"targets_h{h}.parquet")
    p = f.merge(t, on=["ticker", "origin_date"], how="inner", validate="one_to_one")
    p["log_var"] = np.log(p["yz_var"].where(p["yz_var"] > 0))
    return p[["ticker", "origin_date", "horizon"] + FEATURES + ["yz_var", "log_var", "split", "embargo"]]
