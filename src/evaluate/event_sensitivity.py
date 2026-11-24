"""편입 민감도 — 타깃 기준 구간 배정, 이벤트별 집계 (CLAUDE.md 3장, 7.2, 이슈 #50 규칙 1~3).

구간 배정 (규칙 2): 예측 시점 t의 타깃 기간 t+1 ~ t+h가 구간 [a, b] 안에 완전히 들어가는 예측만 쓴다
    → 예측 시점 a−1 ~ b−h,  가능한 예측 수 n_max = (b−a+1) − h + 1

이벤트 e, 모델 m, 구간 s
    L^E_{e,m,s}: 그 구간 예측 시점들의 편입 종목 손실(QLIKE) 평균, L^C: 같은 날짜·같은 모델의 대조군 평균 손실의 평균
    Sens_{e,m,s} = ln(L^E_{e,m,s} / L^E_{e,m,기준}) − ln(L^C_{e,m,s} / L^C_{e,m,기준})
    주 지표 Sens_{m,s} = 이벤트 평균.  보조: 중앙값, pooled(모든 예측의 손실을 합쳐 평균한 뒤 같은 식 — 이전 방식)
    신뢰구간: 편입일을 복원추출하는 부트스트랩(95% 백분위). 뽑힌 편입일의 이벤트를 모두 넣고 이벤트 평균을 낸다.
0에 가까울수록 편입 충격에 덜 민감하다. 인과효과로 해석하지 않는다 (7.2).

대조군: 그날의 구성종목 중 타깃 기간이 어떤 편입·편출 이벤트 윈도우와도 겹치지 않는 종목 (control_windows).

이벤트 선택 (규칙 1): 윈도우 전체가 테스트 5년 안에 들어가는 편입 이벤트. 이벤트의 모델은 기준 구간의 첫 예측 시점이
속한 연도의 윈도우 모델 하나다 (window). 해를 넘는 부분도 같은 모델의 예측(테스트 다음 해)을 쓴다.

조건 (규칙 3, VARIANTS): 기준 구간은 공통 표본 예측 수가 n_max의 base_min 이상(최소 1개), 나머지 구간은 phase_min개 이상.
공통 표본 예측 수 n = 편입 종목이 평가되고 같은 날 대조군 값도 있는 예측 시점 수.
"""
import numpy as np
import pandas as pd

from src.pipeline.windows import PHASE_ORDER, TEST_YEARS

from .sensitivity import control_flag

VARIANTS = {"main": (0.50, 5), "phase_min_1": (0.50, 1), "base_33": (0.33, 5), "base_any": (0.0, 5)}


def target_phases(phases, calendar, h):
    """구간을 예측 시점 범위로 바꾼다. start = a−1, end = b−h, n_max = 가능한 예측 수 (들어가는 예측이 없으면 0)."""
    out = phases.copy()
    a, b = calendar.get_indexer(phases["start"]), calendar.get_indexer(phases["end"])
    out["n_max"] = np.where(a >= 1, np.maximum(b - a + 2 - h, 0), 0)
    out["start"] = calendar[np.maximum(a - 1, 0)]
    out["end"] = calendar[np.maximum(b - h, 0)]
    return out


def select_events(phases, calendar):
    """편입 이벤트별 윈도우와 제외 이유. 열: event_id, ticker, event_type, effective, window, excluded(빈칸이면 사용)."""
    p = phases[phases["direction"] == "편입"]
    rows = []
    for eid, g in p.groupby("event_id"):
        g = g.set_index("phase")
        first = calendar.get_loc(g.loc["baseline", "start"]) - 1
        year = calendar[first].year if first >= 0 else None
        reasons = set(g["reason"])
        if "달력 시작 전" in reasons or year is None or year < TEST_YEARS[0]:
            excluded = "테스트 기간 전에 시작"
        elif "as_of 이후" in reasons or year > TEST_YEARS[-1]:
            excluded = "테스트 기간 뒤까지 이어짐"
        elif g.loc["post", "end"].year > year + 1:
            excluded = "윈도우가 테스트 다음 해를 넘음"
        else:
            excluded = ""
        rows.append({"event_id": eid, "ticker": g["ticker"].iloc[0], "event_type": g["event_type"].iloc[0],
                     "effective": g.loc["short_shock", "start"], "window": year, "excluded": excluded})
    return pd.DataFrame(rows)


def control_windows(phases, calendar, h):
    """대조군 판정용 구간: 이벤트 윈도우와 타깃 기간(t+1 ~ t+h)이 하루라도 겹치는 예측 시점까지 윈도우에 넣는다.
    구간을 타깃 기준으로 배정하면 첫 예측 시점이 윈도우 전날이라, 이렇게 하지 않으면 편입 종목이 자기 대조군에 든다."""
    return phases.assign(start=calendar[np.maximum(calendar.get_indexer(phases["start"]) - h, 0)])


def event_losses(panel, members, control, shifted, events, loss="qlike"):
    """이벤트 × 구간 × 모델의 공통 표본 예측 수와 손실 합. 열: event_id, phase, model, n, sum_E, sum_C.
    control은 control_windows 결과(모든 편입·편출 이벤트), shifted는 target_phases 결과, events는 계산할 이벤트."""
    keys = panel[["ticker", "origin_date"]].drop_duplicates().reset_index(drop=True)
    keys["ctrl"] = control_flag(keys, members, control).to_numpy()
    x = panel.merge(keys, on=["ticker", "origin_date"])
    ctrl = x[x["ctrl"]].groupby(["model", "origin_date"], as_index=False)[loss].mean().rename(columns={loss: "L_C"})
    s = shifted[shifted["event_id"].isin(events["event_id"]) & (shifted["n_max"] > 0)]
    e = s[["event_id", "ticker", "phase", "start", "end"]].merge(
        panel[["ticker", "origin_date", "model", loss]], on="ticker")
    e = e[(e["origin_date"] >= e["start"]) & (e["origin_date"] <= e["end"])].merge(ctrl, on=["model", "origin_date"])
    return e.groupby(["event_id", "phase", "model"], as_index=False).agg(
        n=(loss, "size"), sum_E=(loss, "sum"), sum_C=("L_C", "sum"))


def usage(shifted, events, losses, base_min, phase_min):
    """이벤트 × 구간의 사용 여부와 이유. 열: event_id, phase, n, n_max, reason(빈칸이면 사용). 기준 구간 행도 있다."""
    n = losses.groupby(["event_id", "phase"], as_index=False)["n"].min()
    g = (shifted[shifted["event_id"].isin(events["event_id"])][["event_id", "phase", "complete", "n_max"]]
         .merge(n, on=["event_id", "phase"], how="left"))
    g["n"] = g["n"].fillna(0).astype(int)
    g = g.merge(events[["event_id", "excluded"]], on="event_id")
    base = g[g["phase"] == "baseline"].set_index("event_id")
    base_reason = pd.Series("", index=base.index)
    base_reason[base["n"] < base_min * base["n_max"]] = "기준 구간 예측 비율 부족"
    base_reason[base["n"] == 0] = "기준 구간 예측 없음"
    base_reason[~base["complete"]] = "기준 구간 불완전"
    own = np.select([~g["complete"], g["n_max"] == 0, g["n"] < phase_min],
                    ["구간 불완전", "타깃이 구간에 들어가지 않음", "구간 예측 수 부족"], "")
    reason = np.where(g["phase"] == "baseline", "", own)
    reason = np.where(g["event_id"].map(base_reason) != "", g["event_id"].map(base_reason), reason)
    g["reason"] = np.where(g["excluded"] != "", g["excluded"], reason)
    return g[["event_id", "phase", "n", "n_max", "reason"]]


def event_sens(losses, use):
    """사용하는 (이벤트, 구간)의 Sens_{e,m,s}. 열: event_id, phase, model, n, n_base, 손실 합들, sens."""
    ok = use[use["reason"] == ""][["event_id", "phase"]]
    x = losses.merge(ok, on=["event_id", "phase"])
    base = x[x["phase"] == "baseline"].drop(columns="phase")
    x = x[x["phase"] != "baseline"].merge(base, on=["event_id", "model"], suffixes=("", "_base"))
    x["sens"] = (np.log(x["sum_E"] / x["n"]) - np.log(x["sum_E_base"] / x["n_base"])
                 - np.log(x["sum_C"] / x["n"]) + np.log(x["sum_C_base"] / x["n_base"]))
    return x


def bootstrap_ci(values, clusters, B=1000, seed=0):
    """편입일 단위 복원추출. values: 이벤트 × 모델, clusters: 이벤트의 편입일. 돌려주는 것: (하한, 상한) 모델별 95%."""
    codes, uniq = pd.factorize(clusters)
    G = len(uniq)
    if G < 2:
        nan = np.full(values.shape[1], np.nan)
        return nan, nan
    S = np.zeros((G, values.shape[1]))
    np.add.at(S, codes, values)
    size = np.bincount(codes, minlength=G)
    draw = np.random.default_rng(seed).integers(0, G, size=(B, G))
    W = np.stack([np.bincount(d, minlength=G) for d in draw])
    means = (W @ S) / (W @ size)[:, None]
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def aggregate(es, events, B=1000, seed=0):
    """Sens_{e,m,s} → 모델 × 구간 표. 열: phase, model, n_events, n_effective_dates, sens(주 지표: 이벤트 평균),
    ci_low, ci_high, median, pooled, loo_min, loo_max(이벤트를 하나씩 뺐을 때 평균의 범위)."""
    eff = events.set_index("event_id")["effective"]
    rows = []
    for phase in PHASE_ORDER[1:]:
        g = es[es["phase"] == phase]
        if g.empty:
            continue
        v = g.pivot(index="event_id", columns="model", values="sens").sort_index()
        a = v.to_numpy()
        lo, hi = bootstrap_ci(a, eff.loc[v.index].to_numpy(), B, seed)
        n = len(a)
        loo = (a.sum(axis=0) - a) / (n - 1) if n > 1 else np.full(a.shape, np.nan)
        t = g.groupby("model")[["n", "sum_E", "sum_C", "n_base", "sum_E_base", "sum_C_base"]].sum()
        pooled = (np.log(t["sum_E"] / t["n"]) - np.log(t["sum_E_base"] / t["n_base"])
                  - np.log(t["sum_C"] / t["n"]) + np.log(t["sum_C_base"] / t["n_base"]))
        for j, m in enumerate(v.columns):
            rows.append({"phase": phase, "model": m, "n_events": n, "n_effective_dates": eff.loc[v.index].nunique(),
                         "sens": a[:, j].mean(), "ci_low": lo[j], "ci_high": hi[j], "median": np.median(a[:, j]),
                         "pooled": pooled[m], "loo_min": loo[:, j].min(), "loo_max": loo[:, j].max()})
    return pd.DataFrame(rows)


def leave_one_out(es):
    """이벤트를 하나씩 뺐을 때 이벤트 평균의 변화. 열: event_id, phase, model, sens, mean_without, change."""
    g = es.groupby(["phase", "model"])["sens"]
    total, n = g.transform("sum"), g.transform("size")
    out = es[["event_id", "phase", "model", "sens"]].copy()
    out["mean_without"] = ((total - es["sens"]) / (n - 1)).where(n > 1)
    out["change"] = out["mean_without"] - total / n
    return out.sort_values("change", key=np.abs, ascending=False, ignore_index=True)


def no_baseline_post(losses, use, phase_min=5):
    """기준 구간 조건으로 빠진 이벤트의 사후 구간 ln(L^E / L^C) (이벤트 평균·중앙값, CLAUDE.md 7.2)."""
    out = set(use.loc[use["reason"].str.startswith("기준 구간"), "event_id"])
    x = losses[losses["event_id"].isin(out) & (losses["phase"] == "post") & (losses["n"] >= phase_min)]
    x = x.assign(log_ratio=np.log(x["sum_E"] / x["sum_C"]))
    return x.groupby("model", as_index=False).agg(n_events=("event_id", "nunique"), mean=("log_ratio", "mean"),
                                                  median=("log_ratio", "median"))
