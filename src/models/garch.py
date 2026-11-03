"""GARCH(1,1) — 종목별 종가 수익률로 추정, 다음 h일 평균 조건부 분산 (CLAUDE.md 6.1, 2단계).

    r_t = μ + ε_t,   σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}           (Bollerslev, 1986)
    분산 목표화: ω = s²·(1 − α − β),  s² = 학습 구간 ε의 표본분산
    추정: 학습 구간 수익률의 가우시안 준최대우도 (α, β ≥ 0, α + β < 1), scipy L-BFGS-B
    예측 (t일 종가까지 관측): σ²_{t+1} = ω + α·ε²_t + β·σ²_t
         E[σ²_{t+k}] = s² + (α+β)^{k−1}·(σ²_{t+1} − s²),   예측 = ln( (1/h)·Σ_{k=1}^{h} E[σ²_{t+k}] )

- r_t = 특성 패널의 ret (종가 수익률). 결측일(거래정지·기업행사일)은 추정에서 빼고, 예측 재귀에서는 ε²_t 대신
  σ²_t(조건부 기댓값)를 넣어 갱신한다 (보간 아님 — 정보 없음으로 처리).
- mean: "zero"면 μ = 0, "demean"이면 μ = 학습 구간 평균.
- 학습 구간 관측이 min_obs보다 적은 종목은 다른 종목 추정치(α, β)의 중앙값을 쓰고, s²는 그 종목의 처음
  warmup개 관측의 분산으로 둔다. warmup개가 모이기 전에는 예측하지 않는다 — 예측 시점 이전 정보만 쓴다.
- 목표(Yang–Zhang 분산)와 달리 GARCH는 종가 수익률 분산을 예측한다. 둘 다 일 단위 분산이라 단위는 같다.
"""
import numpy as np
import pandas as pd

PARAMS = {"mean": "zero", "min_obs": 250, "warmup": 60}


def _nll(theta, e2, s2):
    """가우시안 음의 로그우도 (상수 제외). theta = (α, β)."""
    from scipy.signal import lfilter

    a, b = theta
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    w = s2 * (1 - a - b)
    # σ²_t = w + a·e²_{t−1} + b·σ²_{t−1},  σ²_0 = s²
    x = np.concatenate([[s2], w + a * e2[:-1]])
    sig2 = np.maximum(lfilter([1.0], [1.0, -b], x), 1e-12)  # y_n = x_n + b·y_{n−1}
    return 0.5 * np.sum(np.log(sig2) + e2 / sig2)


def fit_one(r, min_obs):
    """학습 구간 수익률(결측 제외) → (α, β, s², μ). 관측이 부족하면 None."""
    from scipy.optimize import minimize

    r = r[np.isfinite(r)]
    if len(r) < min_obs:
        return None
    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.80), (0.03, 0.95)]:
        res = minimize(_nll, [a0, b0], args=(r ** 2, r.var()), method="L-BFGS-B",
                       bounds=[(1e-6, 0.5), (1e-6, 0.998)])
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def fit_predict(panel, h, params=None):
    """패널(load_panel) → (예측 log_var Series, info)."""
    p = {**PARAMS, **(params or {})}
    panel = panel.sort_values(["ticker", "origin_date"])
    ret = panel["ret"].to_numpy(dtype=float)
    is_tr = ((panel["split"] == "train") & ~panel["embargo"]).to_numpy()
    tickers = panel["ticker"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, tickers[1:] != tickers[:-1], True])

    groups, fitted = [], {}
    for s, e in zip(bounds[:-1], bounds[1:]):
        r_tr = ret[s:e][is_tr[s:e]]
        mu = float(np.nanmean(r_tr)) if p["mean"] == "demean" and np.isfinite(r_tr).any() else 0.0
        theta = fit_one(r_tr - mu, p["min_obs"])
        groups.append((s, e, mu, r_tr - mu))
        if theta is not None:
            fitted[s] = theta
    med = np.median(np.array(list(fitted.values())), axis=0) if fitted else np.array([0.05, 0.90])

    out = np.full(len(panel), np.nan)
    for s, e, mu, eps_tr in groups:
        a, b = fitted.get(s, med)
        eps = ret[s:e] - mu
        ok = np.isfinite(eps_tr)
        if ok.sum() >= p["min_obs"]:
            s2 = float(np.var(eps_tr[ok]))
        else:  # 학습 이력이 짧은 종목: 처음 warmup개 관측의 분산 — 그 전에는 예측하지 않음
            seen = np.flatnonzero(np.isfinite(eps))
            if len(seen) < p["warmup"]:
                continue
            first = seen[: p["warmup"]]
            s2 = float(np.var(eps[first]))
            eps = eps.copy()
            eps[: first[-1]] = np.nan  # 분산을 정한 관측 이전에는 예측하지 않음 (아래 start)
        start = int(np.flatnonzero(np.isfinite(eps))[0]) if np.isfinite(eps).any() else len(eps)
        w = s2 * (1 - a - b)
        f = np.mean((a + b) ** np.arange(h))  # (1/h)·Σ_{k=1}^{h} (α+β)^{k−1}
        sig2 = s2
        for t in range(start, len(eps)):
            e2 = eps[t] ** 2 if np.isfinite(eps[t]) else sig2
            sig2 = w + a * e2 + b * sig2  # σ²_{t+1}
            out[s + t] = np.log(s2 + f * (sig2 - s2))
    pred = pd.Series(out, index=panel.index)
    info = {**p, "n_fitted": len(fitted), "n_tickers": len(groups), "median_alpha_beta": med.tolist()}
    return pred, info
