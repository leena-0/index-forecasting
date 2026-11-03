"""모델 간 비교 검정 (CLAUDE.md 7.3).

검정은 날짜별 손실 시계열로 한다: 날짜 t마다 그날 평가된 종목들의 평균 손실 L_{m,t}.
(loss_panel(common=True)를 거치면 모든 모델의 날짜별 종목 집합이 같다.)

Diebold–Mariano (Diebold & Mariano, 1995)
    d_t = L_{1,t} − L_{2,t},   DM = d̄ / sqrt(Ω̂ / T)
    Ω̂ = γ̂_0 + 2 Σ_{j=1}^{q} (1 − j/(q+1)) γ̂_j      Newey–West(Bartlett) 장기분산, q = h − 1
    h일 예측은 구간이 겹쳐 d_t가 h−1차까지 자기상관을 가진다. p값은 표준정규 양측.
    DM < 0이면 모델 1의 손실이 작다.

Model Confidence Set (Hansen, Lunde & Nason, 2011), T_max 통계량
    d_{i·,t} = L_{i,t} − (1/M) Σ_j L_{j,t},   t_i = d̄_{i·} / sqrt(Var̂(d̄_{i·})),   T_max = max_i t_i
    Var̂와 귀무분포는 원형 블록 부트스트랩(블록 길이 기본 h, 같은 부트스트랩 표본을 모든 단계에 재사용):
        T*_max = max_i (d̄*_{i·} − d̄_{i·}) / sqrt(Var̂(d̄_{i·}))
    p = P(T*_max ≥ T_max). p < α이면 t_i가 가장 큰 모델을 빼고 반복한다.
    MCS p값 = 그 모델이 빠질 때까지의 p값의 누적 최댓값. 남은 모델(MCS p ≥ α)이 신뢰집합이다.
"""
import math

import numpy as np
import pandas as pd


def daily_losses(panel, loss="qlike"):
    """손실 패널 → 날짜 × 모델 표 (그날 종목 평균). 모든 모델에 값이 있는 날짜만 남긴다."""
    if panel["horizon"].nunique() != 1 or panel["split"].nunique() != 1:
        raise ValueError("한 분할·한 예측 기간의 패널만 넣으세요.")
    return panel.pivot_table(index="origin_date", columns="model", values=loss, aggfunc="mean").dropna()


def newey_west_var(d, lag):
    """Ω̂ = γ̂_0 + 2 Σ_{j=1}^{lag} (1 − j/(lag+1)) γ̂_j,  γ̂_j = (1/T) Σ_t (d_t − d̄)(d_{t−j} − d̄)."""
    d = np.asarray(d, dtype=float)
    e = d - d.mean()
    T = len(e)
    omega = e @ e / T
    for j in range(1, min(lag, T - 1) + 1):
        omega += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / T
    return omega


def dm_test(loss1, loss2, h):
    """Diebold–Mariano 검정. 돌려주는 것: dict(mean_diff, stat, p_value, n)."""
    d = np.asarray(loss1, dtype=float) - np.asarray(loss2, dtype=float)
    T = len(d)
    omega = newey_west_var(d, h - 1)
    stat = d.mean() / math.sqrt(omega / T) if omega > 0 else float("nan")
    p = math.erfc(abs(stat) / math.sqrt(2)) if np.isfinite(stat) else float("nan")
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p_value": float(p), "n": T}


def dm_table(daily, h):
    """daily_losses 결과의 모든 모델 쌍에 DM 검정. 열: model_1, model_2, mean_diff, stat, p_value, n."""
    models = list(daily.columns)
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            rows.append({"model_1": a, "model_2": b, **dm_test(daily[a], daily[b], h)})
    return pd.DataFrame(rows)


def block_bootstrap_index(T, block, B, rng):
    """원형 블록 부트스트랩 인덱스 (B × T). 블록 시작점을 균등하게 뽑아 길이 block씩 이어 붙인다."""
    n_blocks = -(-T // block)
    starts = rng.integers(0, T, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
    return idx.reshape(B, -1)[:, :T]


def mcs(daily, alpha=0.10, B=1000, block=None, seed=0):
    """Model Confidence Set (T_max). daily: 날짜 × 모델 손실 표.
    block: 블록 길이 (None이면 1). 보통 예측 기간 h를 넘긴다.
    돌려주는 것: DataFrame[model, mean_loss, eliminated_order, mcs_p, in_mcs] (eliminated_order: 빠진 순서,
    마지막까지 남은 모델은 NaN)."""
    L = daily.to_numpy(dtype=float)
    T, M = L.shape
    idx = block_bootstrap_index(T, block or 1, B, np.random.default_rng(seed))
    boot_means = L[idx].mean(axis=1)  # B × M: 부트스트랩 표본의 모델별 평균 손실
    mean = L.mean(axis=0)
    alive = list(range(M))
    order, p_running, out = 0, 0.0, {}
    while len(alive) > 1:
        a = np.array(alive)
        dbar = mean[a] - mean[a].mean()
        dstar = boot_means[:, a] - boot_means[:, a].mean(axis=1, keepdims=True)
        var = ((dstar - dbar) ** 2).mean(axis=0)
        t = dbar / np.sqrt(var)
        tmax_star = ((dstar - dbar) / np.sqrt(var)).max(axis=1)
        p = float((tmax_star >= t.max()).mean())
        p_running = max(p_running, p)
        if p >= alpha:
            break
        worst = alive[int(np.argmax(t))]
        order += 1
        out[worst] = (order, p_running)
        alive.remove(worst)
    for m in alive:
        out[m] = (np.nan, 1.0 if len(alive) == 1 else p_running)
    res = pd.DataFrame({"model": daily.columns, "mean_loss": mean,
                        "eliminated_order": [out[i][0] for i in range(M)],
                        "mcs_p": [out[i][1] for i in range(M)]})
    res["in_mcs"] = res["mcs_p"] >= alpha
    return res.sort_values("mean_loss", ignore_index=True)
