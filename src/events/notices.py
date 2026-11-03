"""KRX 지수 공지 목록(공지 단위) → 공지일 CSV(종목 단위, docs/EVENTS.md 형식).

공지 목록 열: 변경적용일, 변경유형, 링크, 게시물번호, 공지일, 제목, 비고 (UTF-8, BOM 가능).
공지 목록에는 종목코드가 없으므로, 구성종목 변화에서 만든 이벤트와 적용일로 짝지어 종목 단위 레코드를 만든다.

짝짓기 규칙
- 반영된 공지만 쓴다: 변경적용일이 있고, 변경유형에 괄호 표시(취소·미실시·일정변경)가 없고, 비고에 "미반영"이 없는 공지.
- 이벤트의 effective_date와 공지의 변경적용일이 같으면 짝이다. 같은 적용일의 공지가 여럿이면 가장 이른 공지일을 쓴다
  (더 이른 공지일은 "공지일 이전 정보만" 규칙(CLAUDE.md 3장)에서 보수적이다).
- 짝이 없는 이벤트는 공지일 CSV에 넣지 않고 따로 돌려준다 → 파이프라인의 일대일 검사(reconcile_events)에서 공지 누락으로 나온다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .loader import KEY_COLUMNS, REQUIRED_COLUMNS, EventDataError

NOTICE_COLUMNS = {
    "변경적용일": "effective_date",
    "변경유형": "notice_type",
    "링크": "source_url",
    "게시물번호": "notice_id",
    "공지일": "announce_date",
    "제목": "title",
    "비고": "note",
}


def load_notices(path: str | Path) -> pd.DataFrame:
    """공지 목록 CSV를 읽고 반영 여부(applied)를 붙인다."""
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string").rename(columns=NOTICE_COLUMNS)
    missing = [column for column in NOTICE_COLUMNS.values() if column not in frame]
    if missing:
        raise EventDataError(f"공지 목록에 열이 없습니다: {missing}")
    for column in ["effective_date", "announce_date"]:
        frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d")
    frame["applied"] = (
        frame["effective_date"].notna()
        & ~frame["notice_type"].str.contains(r"\(", na=False)
        & ~frame["note"].str.contains("미반영", na=False)
    )
    bad = frame["applied"] & (frame["announce_date"] >= frame["effective_date"])
    if bad.any():
        raise EventDataError(f"공지일이 변경적용일보다 늦거나 같은 공지: {frame.loc[bad, 'notice_id'].tolist()}")
    return frame


def notices_to_announcements(
    events: pd.DataFrame, notices: pd.DataFrame, source_type: str, collected_at: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """이벤트(ticker, event_type, direction, effective_date)와 공지 목록 → (공지일 CSV 행, 짝 없는 이벤트, 이벤트 없는 공지)."""
    applied = notices[notices["applied"]].sort_values(["announce_date", "notice_id"])
    first = applied.groupby("effective_date").first()
    events = events[KEY_COLUMNS]  # 임시 공지일(announce_date) 등 다른 열은 쓰지 않는다
    hit = events["effective_date"].isin(first.index)
    matched = events[hit].join(first[["announce_date", "source_url"]], on="effective_date")
    rows = matched.assign(
        announce_date=matched["announce_date"].dt.strftime("%Y-%m-%d"),
        effective_date=matched["effective_date"].dt.strftime("%Y-%m-%d"),
        source_type=source_type,
        collected_at=collected_at,
    )[REQUIRED_COLUMNS]
    rows = rows.sort_values(["effective_date", "ticker", "direction"], ignore_index=True)
    unmatched = events[~hit].reset_index(drop=True)
    unused = applied[~applied["effective_date"].isin(events["effective_date"])].reset_index(drop=True)
    return rows, unmatched, unused
