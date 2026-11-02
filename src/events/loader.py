"""편입·편출 공지일 CSV를 공통 형식으로 읽고 구성종목 변화와 대조한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EVENT_TYPES = {"정기변경", "수시변경", "특례편입"}
DIRECTIONS = {"편입", "편출"}
SOURCE_TYPES = {"manual", "crawled"}
KEY_COLUMNS = ["ticker", "event_type", "direction", "effective_date"]
REQUIRED_COLUMNS = KEY_COLUMNS + [
    "announce_date",
    "source_url",
    "source_type",
    "collected_at",
]


class EventDataError(ValueError):
    """이벤트 파일의 형식이나 결합 관계가 올바르지 않음."""


@dataclass(frozen=True)
class Reconciliation:
    events: pd.DataFrame
    missing: pd.DataFrame
    unexpected: pd.DataFrame

    @property
    def complete(self) -> bool:
        return self.missing.empty and self.unexpected.empty

    def require_complete(self) -> None:
        if self.complete:
            return
        missing = self.missing[KEY_COLUMNS].head(5).to_dict("records")
        unexpected = self.unexpected[KEY_COLUMNS].head(5).to_dict("records")
        raise EventDataError(
            "구성종목 변화와 공지일 데이터가 일치하지 않습니다: "
            f"공지 누락 {len(self.missing)}건 {missing}, "
            f"대응하는 구성종목 변화 없음 {len(self.unexpected)}건 {unexpected}"
        )


def _required_text(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        rows = frame.index[values.isna() | values.eq("")].tolist()[:5]
        raise EventDataError(f"{column}이 비어 있는 행: {rows}")
    return values


def _date(values: pd.Series, column: str) -> pd.Series:
    if not values.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise EventDataError(f"{column}은 YYYY-MM-DD 형식이어야 합니다.")
    try:
        return pd.to_datetime(values, format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as exc:
        raise EventDataError(f"{column}에 유효하지 않은 날짜가 있습니다.") from exc


def load_announcements(path: str | Path, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """공지일 CSV를 읽고 값·날짜·중복을 검증한다."""
    path = Path(path)
    if not path.is_file():
        raise EventDataError(f"이벤트 파일이 없습니다: {path}")
    frame = pd.read_csv(path, dtype="string")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing_columns:
        raise EventDataError(f"필수 열이 없습니다: {missing_columns}")
    if frame.empty:
        raise EventDataError("이벤트 파일이 비어 있습니다.")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = _required_text(frame, column)

    numeric_ticker = frame["ticker"].str.fullmatch(r"\d{1,6}")
    if not numeric_ticker.all():
        bad = frame.loc[~numeric_ticker, "ticker"].tolist()[:5]
        raise EventDataError(f"종목코드는 1~6자리 숫자여야 합니다: {bad}")
    frame["ticker"] = frame["ticker"].str.zfill(6)

    for column, allowed in (
        ("event_type", EVENT_TYPES),
        ("direction", DIRECTIONS),
        ("source_type", SOURCE_TYPES),
    ):
        unknown = sorted(set(frame[column]) - allowed)
        if unknown:
            raise EventDataError(f"알 수 없는 {column}: {unknown}")

    frame["announce_date"] = _date(frame["announce_date"], "announce_date")
    frame["effective_date"] = _date(frame["effective_date"], "effective_date")
    bad_order = frame["announce_date"] >= frame["effective_date"]
    if bad_order.any():
        rows = frame.index[bad_order].tolist()[:5]
        raise EventDataError(f"announce_date는 effective_date보다 빨라야 합니다. 행: {rows}")
    try:
        pd.to_datetime(frame["collected_at"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise EventDataError("collected_at에 유효하지 않은 날짜·시간이 있습니다.") from exc

    duplicate = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicate.any():
        keys = frame.loc[duplicate, KEY_COLUMNS].head(5).to_dict("records")
        raise EventDataError(f"이벤트 키가 중복됩니다: {keys}")

    if as_of is not None:
        frame = frame[frame["effective_date"] <= pd.Timestamp(as_of)].copy()
    return frame.sort_values(["effective_date", "ticker", "direction"], ignore_index=True)


def reconcile_events(expected: pd.DataFrame, announcements: pd.DataFrame) -> Reconciliation:
    """구성종목 변화에서 만든 이벤트마다 공지 레코드가 정확히 하나인지 대조한다."""
    missing_expected = [column for column in ["event_id", *KEY_COLUMNS] if column not in expected]
    missing_announcements = [column for column in REQUIRED_COLUMNS if column not in announcements]
    if missing_expected:
        raise EventDataError(f"구성종목 이벤트 필수 열이 없습니다: {missing_expected}")
    if missing_announcements:
        raise EventDataError(f"공지일 데이터 필수 열이 없습니다: {missing_announcements}")
    if expected.duplicated(KEY_COLUMNS).any():
        raise EventDataError("구성종목 이벤트 키가 중복됩니다.")
    if announcements.duplicated(KEY_COLUMNS).any():
        raise EventDataError("공지일 이벤트 키가 중복됩니다.")

    expected_keys = expected[KEY_COLUMNS].copy()
    announcement_keys = announcements[KEY_COLUMNS].copy()
    missing = expected_keys.merge(announcement_keys, on=KEY_COLUMNS, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    unexpected = announcement_keys.merge(expected_keys, on=KEY_COLUMNS, how="left", indicator=True)
    unexpected = unexpected[unexpected["_merge"] == "left_only"].drop(columns="_merge")

    details = announcements.rename(
        columns={
            "announce_date": "actual_announce_date",
            "source_type": "actual_announce_source",
        }
    )
    detail_columns = KEY_COLUMNS + [
        "actual_announce_date",
        "source_url",
        "actual_announce_source",
        "collected_at",
    ]
    merged = expected.merge(details[detail_columns], on=KEY_COLUMNS, how="inner", validate="one_to_one")
    merged["announce_date"] = merged.pop("actual_announce_date")
    merged["announce_source"] = merged.pop("actual_announce_source")
    first = [
        "event_id",
        "ticker",
        "event_type",
        "direction",
        "announce_date",
        "effective_date",
        "announce_source",
        "source_url",
        "collected_at",
    ]
    merged = merged[first].sort_values(["effective_date", "ticker"], ignore_index=True)
    return Reconciliation(events=merged, missing=missing, unexpected=unexpected)
