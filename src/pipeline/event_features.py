"""이벤트 단위 변수 (CLAUDE.md 6.2, 이슈 #50 규칙 4). t일 변수는 t일 전에 공지된 정보만 쓴다.

값이 있는 기간: 공지 다음 거래일 ~ 편입(편출)일 + 21영업일. 공지 시각이 장 마감 전인지 모르므로 공지 당일에는 쓰지 않는다.
종목·거래일(t)마다
    event_active = 위 기간이면 1, 아니면 0
    event_dir    = 편입 +1, 편출 −1, 없음 0
    event_type   = 정기변경 1, 수시·특례 0
    event_dist   = t − 적용일 (영업일). 적용 전 −, 당일 0, 적용 후 + (최대 +21), 없음 0
    event_n      = ln(1 + n), n = 같은 적용일에 바뀌는 종목 수 (t일 전에 공지된 편입·편출 종목만), 없음 0

- 같은 종목의 이벤트 기간이 겹치면 더 최근에 공지된 이벤트를 쓴다.
- known_from: 적용일이 나중 공지에서 확정됐거나 같은 적용일에 공지가 여러 건인 경우, 그 적용일의 변수를 쓰기 시작할
  공지일(설정 event_known_from). 공지일보다 빠르면 공지일을 쓴다. 5구간의 공지일은 바꾸지 않는다.
- 한계: 공지됐다가 실시되지 않은 변경은 이벤트 자료에 없어 변수도 없다 (예: 2021-05 서부T&D, docs/PIPELINE.md).
"""
import numpy as np
import pandas as pd

EVENT_FEATURES = ["event_active", "event_dir", "event_type", "event_dist", "event_n"]
POST_DAYS = 21


def build_event_features(keys, events, calendar, known_from=None):
    """keys: DataFrame[ticker, date], events: DataFrame[event_id, ticker, event_type, direction, announce_date,
    effective_date]. 돌려주는 것: DataFrame[ticker, date] + EVENT_FEATURES (keys와 같은 순서)."""
    e = events[["event_id", "ticker", "event_type", "direction", "announce_date", "effective_date"]].copy()
    known = pd.to_datetime(e["announce_date"])
    if known_from:
        override = pd.to_datetime(e["effective_date"].map({pd.Timestamp(k): v for k, v in known_from.items()}))
        known = known.where(override.isna() | (override < known), override)
    e["known"] = known
    e["start"] = calendar.searchsorted(e["known"], side="right")
    e["eff"] = calendar.searchsorted(pd.to_datetime(e["effective_date"]))
    e["end"] = np.minimum(e["eff"] + POST_DAYS, len(calendar) - 1)
    rows = []
    for r in e.sort_values(["known", "effective_date", "event_id"]).itertuples():
        if r.start > r.end:
            continue
        days = np.arange(r.start, r.end + 1)
        starts = e.loc[e["effective_date"] == r.effective_date].drop_duplicates("ticker")["start"].to_numpy()
        rows.append(pd.DataFrame({
            "ticker": r.ticker, "date": calendar[days], "event_active": 1.0,
            "event_dir": 1.0 if r.direction == "편입" else -1.0, "event_type": float(r.event_type == "정기변경"),
            "event_dist": (days - r.eff).astype(float), "event_n": np.log1p((starts[:, None] <= days).sum(axis=0))}))
    out = keys[["ticker", "date"]]
    if not rows:
        return out.assign(**{c: 0.0 for c in EVENT_FEATURES})
    active = pd.concat(rows, ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")  # 최근 공지가 남는다
    out = out.merge(active, on=["ticker", "date"], how="left")
    out[EVENT_FEATURES] = out[EVENT_FEATURES].fillna(0.0)
    return out


def event_matrix(panel):
    """신경망 입력용 이벤트 변수 행렬 (n × 5, 변수가 없는 패널이면 n × 0). 거리는 21로 나눈다.
    드물게만 0이 아닌 변수라 표준화하지 않는다 (표준편차가 작아 값이 지나치게 커진다)."""
    cols = [c for c in EVENT_FEATURES if c in panel]
    x = panel[cols].to_numpy(dtype=np.float32).copy()
    if "event_dist" in cols:
        x[:, cols.index("event_dist")] /= POST_DAYS
    return x
