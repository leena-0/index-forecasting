"""편입 민감도 — 이중차분 (CLAUDE.md 7.2).

모델 m, 구간 s에 대해
    Sens_{m,s} = ln(L^E_{m,s} / L^E_{m,기준}) − ln(L^C_{m,s} / L^C_{m,기준})
    L^E: 편입 종목의 평균 손실(기본 QLIKE). 이벤트 × 날짜를 모두 모은 평균
    L^C: 같은 날짜의 대조군 평균 손실. 날짜마다 대조군 종목 평균을 내고, 편입 종목이 평가된 날짜들로 평균
0에 가까울수록 편입 충격에 덜 민감하다.

- 이벤트: event_phases에서 direction이 편입이고, 윈도우가 분할 경계를 넘지 않으며(crosses_split=False),
  기준 구간과 구간 s가 모두 완전한(complete) 것만 Sens_{m,s}에 들어간다.
- 대조군(기존 구성종목): 그날의 구성종목 중, 그날 어떤 편입·편출 이벤트 윈도우(첫 구간 시작 ~ 마지막 구간 끝)
  안에도 있지 않은 종목.
- 기준 구간이 없는(불완전한) 편입 이벤트(특례편입 등)는 Sens에서 빠지고, 사후 구간의
  ln(L^E_사후 / L^C_사후)를 따로 계산한다 (no_baseline_post).
"""
import numpy as np
import pandas as pd


def control_flag(panel, members, phases):
    """panel 행마다 대조군 여부 (bool Series). members: DataFrame[date, ticker]."""
    m = members.rename(columns={"date": "origin_date"}).assign(member=True)
    is_member = panel[["origin_date", "ticker"]].merge(m, on=["origin_date", "ticker"], how="left")["member"]
    win = phases.groupby(["event_id", "ticker"], as_index=False).agg(w_start=("start", "min"), w_end=("end", "max"))
    x = panel[["ticker", "origin_date"]].reset_index().merge(win, on="ticker")
    in_window = x.loc[(x["origin_date"] >= x["w_start"]) & (x["origin_date"] <= x["w_end"]), "index"].unique()
    flag = is_member.fillna(False).astype(bool).to_numpy().copy()
    flag[np.isin(panel.index.to_numpy(), in_window)] = False
    return pd.Series(flag, index=panel.index)


def event_control_losses(panel, members, phases, loss="qlike"):
    """편입 이벤트 × 구간 × 모델의 손실 합계. 열: event_id, phase, split, model, complete, n, sum_E, sum_C.
    n = 편입 종목이 평가되고 같은 날 대조군 값도 있는 날짜 수, sum_E/sum_C = 그 날짜들의 손실 합."""
    panel = panel.reset_index(drop=True)
    ctrl = (panel[control_flag(panel, members, phases)]
            .groupby(["model", "origin_date"], as_index=False)[loss].mean().rename(columns={loss: "L_C"}))
    ev = phases[(phases["direction"] == "편입") & ~phases["crosses_split"]]
    x = ev[["event_id", "ticker", "phase", "split", "start", "end", "complete"]].merge(
        panel[["ticker", "origin_date", "split", "model", loss]], on=["ticker", "split"])
    x = x[(x["origin_date"] >= x["start"]) & (x["origin_date"] <= x["end"])]
    x = x.merge(ctrl, on=["model", "origin_date"])
    return (x.groupby(["event_id", "phase", "split", "model", "complete"], as_index=False)
             .agg(n=(loss, "size"), sum_E=(loss, "sum"), sum_C=("L_C", "sum")))


def sensitivity(ec, baseline="baseline"):
    """event_control_losses 결과 → Sens_{m,s}. 열: split, model, phase, n_events, L_E, L_C, L_E_base, L_C_base, sens.
    구간 s마다 기준 구간과 s가 모두 완전한 이벤트만 모은다."""
    c = ec[ec["complete"]]
    base = c[c["phase"] == baseline][["event_id", "model", "n", "sum_E", "sum_C"]]
    x = c[c["phase"] != baseline].merge(base, on=["event_id", "model"], suffixes=("", "_base"))
    g = x.groupby(["split", "model", "phase"], as_index=False).agg(
        n_events=("event_id", "nunique"), n=("n", "sum"), sum_E=("sum_E", "sum"), sum_C=("sum_C", "sum"),
        n_base=("n_base", "sum"), sum_E_base=("sum_E_base", "sum"), sum_C_base=("sum_C_base", "sum"))
    g["L_E"], g["L_C"] = g["sum_E"] / g["n"], g["sum_C"] / g["n"]
    g["L_E_base"], g["L_C_base"] = g["sum_E_base"] / g["n_base"], g["sum_C_base"] / g["n_base"]
    g["sens"] = np.log(g["L_E"] / g["L_E_base"]) - np.log(g["L_C"] / g["L_C_base"])
    return g[["split", "model", "phase", "n_events", "L_E", "L_C", "L_E_base", "L_C_base", "sens"]]


def no_baseline_post(ec, baseline="baseline", post="post"):
    """기준 구간이 불완전한 편입 이벤트의 사후 구간 ln(L^E / L^C). 열: split, model, n_events, L_E, L_C, log_ratio."""
    has_base = set(ec.loc[(ec["phase"] == baseline) & ec["complete"], "event_id"])
    x = ec[(ec["phase"] == post) & ec["complete"] & ~ec["event_id"].isin(has_base)]
    g = x.groupby(["split", "model"], as_index=False).agg(
        n_events=("event_id", "nunique"), n=("n", "sum"), sum_E=("sum_E", "sum"), sum_C=("sum_C", "sum"))
    g["L_E"], g["L_C"] = g["sum_E"] / g["n"], g["sum_C"] / g["n"]
    g["log_ratio"] = np.log(g["L_E"] / g["L_C"])
    return g[["split", "model", "n_events", "L_E", "L_C", "log_ratio"]]


def event_window_rows(panel, phases):
    """panel 행이 편입 이벤트 윈도우 안인지 (bool Series, panel과 같은 index). 7.4 예측 기간 선택용.
    윈도우 = 편입 이벤트(crosses_split=False)의 완전한(complete) 구간들 [start, end], 같은 분할 안."""
    ev = phases[(phases["direction"] == "편입") & ~phases["crosses_split"] & phases["complete"]]
    x = panel[["ticker", "origin_date", "split"]].reset_index().merge(
        ev[["ticker", "split", "start", "end"]], on=["ticker", "split"])
    idx = x.loc[(x["origin_date"] >= x["start"]) & (x["origin_date"] <= x["end"]), "index"].unique()
    return pd.Series(panel.index.isin(idx), index=panel.index)
