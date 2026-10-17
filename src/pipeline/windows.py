"""분할·경계 공백·편입 전후 구간 배정 (CLAUDE.md 3장, 4.1, 4.3).

- 분할: 날짜 ≤ train_end → train, ≤ val_end → val, 그 이후 → test (as_of까지)
- 경계 공백(embargo): 예측은 origin_date의 분할에 배정하고, 타깃 구간 끝(origin + h영업일)이
  다른 분할에 들어가면 제외한다 → 각 경계 직전 h영업일이 비워진다.
- 이벤트는 공지일이 속한 분할에 배정하고, 이벤트 전체 윈도우(첫 구간 시작 ~ 마지막 구간 끝)가
  분할 경계를 넘으면 그 분할 분석에서 제외한다 (crosses_split).
- 구간: config의 event_phases(영업일 오프셋, 양 끝 포함) + announce_to_effective = [공지일, 편입일 직전].
  구간이 다 관찰되지 않으면(달력 밖, 종목 데이터 기간 밖) complete=False로 두고 이유를 남긴다.
"""
import numpy as np
import pandas as pd

PHASE_ORDER = ["baseline", "anticipation", "announce_to_effective", "short_shock", "post"]


def split_of(dates, split_cfg):
    dates = pd.DatetimeIndex(dates)
    train_end, val_end = pd.Timestamp(split_cfg["train_end"]), pd.Timestamp(split_cfg["val_end"])
    return np.where(dates <= train_end, "train", np.where(dates <= val_end, "val", "test"))


def assign_splits(targets, calendar, split_cfg):
    """targets[ticker, origin_date, horizon, ...]에 split, embargo 열을 붙인다.
    embargo=True: 타깃 구간 끝이 다른 분할이거나 달력 밖(as_of 이후)이라 쓰면 안 되는 행."""
    t = targets.copy()
    pos = calendar.get_indexer(t["origin_date"])
    if (pos < 0).any():
        raise ValueError("origin_date가 달력에 없습니다.")
    end_pos = pos + t["horizon"].to_numpy()
    inside = end_pos < len(calendar)
    t["split"] = split_of(t["origin_date"], split_cfg)
    end_split = np.full(len(t), "", dtype=object)
    end_split[inside] = split_of(calendar[end_pos[inside]], split_cfg)
    t["embargo"] = ~inside | (end_split != t["split"].to_numpy())
    return t


def event_phases(events, calendar, prices, phases_cfg, split_cfg):
    """이벤트 × 구간 표. 열: event_id, ticker, event_type, direction, phase, start, end,
    complete, reason, split, crosses_split, baseline_overlaps_prev."""
    span = prices.groupby("ticker")["date"].agg(["min", "max"])
    last = len(calendar) - 1
    rows = []
    for e in events.itertuples():
        a = calendar.searchsorted(e.announce_date)   # 공지일이 휴장일이면 다음 거래일
        f = calendar.searchsorted(e.effective_date)
        ranges = {}
        for name, p in phases_cfg.items():
            base = a if p["anchor"] == "announce" else f
            ranges[name] = (base + p["range"][0], base + p["range"][1])
        ranges["announce_to_effective"] = (a, f - 1)
        lo, hi = span.loc[e.ticker] if e.ticker in span.index else (None, None)
        for name in PHASE_ORDER:
            s, t = ranges[name]
            if t < s:
                reason = "빈 구간" + (" (임시 공지일)" if e.announce_source == "placeholder" else "")
            elif s < 0:
                reason = "달력 시작 전"
            elif t > last:
                reason = "as_of 이후"
            elif lo is None or calendar[s] < lo or calendar[t] > hi:
                reason = "종목 데이터 기간 밖"
            else:
                reason = ""
            rows.append({
                "event_id": e.event_id, "ticker": e.ticker, "event_type": e.event_type,
                "direction": e.direction, "phase": name,
                "start": calendar[min(max(s, 0), last)], "end": calendar[min(max(t, 0), last)],
                "complete": reason == "", "reason": reason,
            })
    out = pd.DataFrame(rows)

    # 분할: 공지일 기준, 전체 윈도우가 경계를 넘는지
    ev = events.set_index("event_id")
    out["split"] = split_of(ev.loc[out["event_id"], "announce_date"], split_cfg)
    w = out.groupby("event_id").agg(w_start=("start", "min"), w_end=("end", "max"))
    w["crosses_split"] = split_of(w["w_start"], split_cfg) != split_of(w["w_end"], split_cfg)
    out = out.merge(w[["crosses_split"]], left_on="event_id", right_index=True)

    # 같은 종목의 직전 이벤트 사후 구간과 기준 구간이 겹치는지 (CLAUDE.md 4.3: 표시해 둔다)
    b = out[out["phase"] == "baseline"][["event_id", "ticker", "start"]]
    p = out[out["phase"] == "post"][["event_id", "ticker", "end"]]
    flags = {}
    for tk, g in b.groupby("ticker"):
        posts = p[p["ticker"] == tk]
        for r in g.itertuples():
            prev = posts[posts["event_id"] < r.event_id]
            flags[r.event_id] = bool((prev["end"] >= r.start).any())
    out["baseline_overlaps_prev"] = out["event_id"].map(flags).fillna(False).astype(bool)
    return out


def summarize(phases):
    """구간별 이벤트 수와 제외 건수 (CLAUDE.md 4.3: 제외 건수를 결과와 함께 보고)."""
    g = phases.groupby(["split", "phase", "reason"]).size().rename("n").reset_index()
    g["phase"] = pd.Categorical(g["phase"], PHASE_ORDER)
    return g.sort_values(["split", "phase", "reason"]).reset_index(drop=True)
