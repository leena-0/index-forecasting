"""EWMA 변동성 (RiskMetrics, J.P. Morgan/Reuters 1996).

    σ²_t = λ·σ²_{t−1} + (1 − λ)·rv_d_t        rv_d: 일별 분산 대용치 (src/pipeline/features.py)
    다음 h일의 일 단위 분산 예측 = σ²_t        (EWMA 예측은 예측 기간과 관계없이 평평하다)

- λ = 0.94 고정이 기본 (RiskMetrics 일별 값). 다른 λ는 검증 구간 비교용.
- 초기값: 종목의 처음 warmup개(기본 22) rv_d 평균. 그 전에는 예측하지 않는다.
- rv_d가 결측인 날은 갱신하지 않고 직전 σ²를 그대로 둔다 (보간 대신 정보 없음으로 처리).
t일 예측은 t일까지의 rv_d만 쓴다.
"""
import numpy as np
import pandas as pd


def ewma_var(rv, lam=0.94, warmup=22):
    """rv_d 배열(한 종목, 거래일 순) → σ²_t 배열. 예측하지 않는 날은 NaN."""
    rv = np.asarray(rv, dtype=float)
    out = np.full(len(rv), np.nan)
    seen = np.flatnonzero(~np.isnan(rv))
    if len(seen) < warmup:
        return out
    start = seen[warmup - 1]
    s = rv[seen[:warmup]].mean()
    out[start] = s
    for t in range(start + 1, len(rv)):
        if not np.isnan(rv[t]):
            s = lam * s + (1 - lam) * rv[t]
        out[t] = s
    return out


def predict(panel, lam=0.94, warmup=22):
    """패널(load_panel) → σ²_t Series (panel과 같은 index)."""
    p = panel.sort_values(["ticker", "origin_date"])
    pred = pd.Series(np.nan, index=p.index)
    for _, g in p.groupby("ticker", sort=False):
        pred.loc[g.index] = ewma_var(g["rv_d"].to_numpy(), lam, warmup)
    return pred.reindex(panel.index)
