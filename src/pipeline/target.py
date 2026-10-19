"""예측 타깃: 다음 h영업일의 Yang–Zhang 분산 (CLAUDE.md 5장).

Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation based on high, low, open,
and close prices. Journal of Business, 73(3), 477–491.

h일 구간(i = 1, …, n, n = h)에 대해
    o_i = ln(O_i / C_{i-1})   야간 수익률 (C_0 = 구간 시작 전날 종가)
    c_i = ln(C_i / O_i)       장중 수익률
    u_i = ln(H_i / O_i),  d_i = ln(L_i / O_i)

    demean=True  (원래 정의):  σo² = 1/(n−1) · Σ(o_i − ō)²,  σc² = 1/(n−1) · Σ(c_i − c̄)²
    demean=False (평균 0 가정): σo² = 1/n · Σ o_i²,          σc² = 1/n · Σ c_i²
    σRS² = 1/n · Σ [u_i(u_i − c_i) + d_i(d_i − c_i)]
    σYZ² = σo² + k·σc² + (1 − k)·σRS²,   k = 0.34 / (1.34 + (n+1)/(n−1))

타깃(origin t) = 구간 t+1 ~ t+h의 σYZ² (일 단위 분산). o_1에는 t의 종가가 들어간다.
구간 안에 결측(거래정지, 상장 전, 데이터 끝)이 하나라도 있으면 NaN — 보간하지 않는다.
거래정지 다음 날은 직전 종가가 없으므로 o가 NaN이 되어, 그날이 들어간 구간도 NaN이다.
prices에 corp_action 열이 있으면(수정 안 된 가격, openapi 어댑터) 그날의 o도 NaN으로 두어
분할·증자로 가격 기준이 바뀐 날이 들어간 구간을 결측으로 처리한다.
"""
import numpy as np
import pandas as pd


def yz_k(n):
    """Yang–Zhang 가중치 k. n = 5 → 약 0.120, n = 22 → 약 0.140."""
    if n < 2:
        raise ValueError("n은 2 이상이어야 합니다.")
    return 0.34 / (1.34 + (n + 1) / (n - 1))


def forward_yz(prices, calendar, h, demean=True):
    """종목·거래일마다 다음 h일의 Yang–Zhang 분산.

    prices: DataFrame[date, ticker, open, high, low, close] (거래정지일은 NaN)
    calendar: 거래일 DatetimeIndex. 종목마다 첫 행 ~ 마지막 행 사이를 달력에 맞춰 펼친 뒤 계산한다
              (달력에 있는데 행이 없는 날은 결측).
    돌려주는 것: DataFrame[ticker, origin_date, horizon, yz_var]
    """
    k = yz_k(h)
    out = []
    for ticker, g in prices.groupby("ticker", sort=True):
        g = g.set_index("date").sort_index()
        days = calendar[(calendar >= g.index[0]) & (calendar <= g.index[-1])]
        g = g.reindex(days)
        o = np.log(g["open"] / g["close"].shift(1))
        if "corp_action" in g:  # 수정 안 된 가격: 가격 기준이 바뀐 날의 야간 수익률은 쓰지 않는다
            o = o.mask(g["corp_action"].fillna(False).astype(bool))
        c = np.log(g["close"] / g["open"])
        u = np.log(g["high"] / g["open"])
        d = np.log(g["low"] / g["open"])
        rs = u * (u - c) + d * (d - c)
        if demean:
            var_o = o.rolling(h, min_periods=h).var(ddof=1)
            var_c = c.rolling(h, min_periods=h).var(ddof=1)
        else:
            var_o = (o ** 2).rolling(h, min_periods=h).mean()
            var_c = (c ** 2).rolling(h, min_periods=h).mean()
        var_rs = rs.rolling(h, min_periods=h).mean()
        yz = var_o + k * var_c + (1 - k) * var_rs  # 행 j = 구간 j−h+1 ~ j
        out.append(pd.DataFrame({"ticker": ticker, "origin_date": days, "horizon": h,
                                 "yz_var": yz.shift(-h).values}))  # origin t → 구간 t+1 ~ t+h
    return pd.concat(out, ignore_index=True)
