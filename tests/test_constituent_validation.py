"""구성종목 스냅샷 검증에서 날짜 중첩과 취소 공지 처리 검증."""

import pandas as pd

from src.constituent_validation import (
    classify_notices, compare_adjacent, compare_experiment_events,
    reconstruct_experiment_membership, validate_prices,
)


def test_consecutive_changes_use_overlapping_snapshots():
    days = list(pd.to_datetime(["2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27"]))
    snapshots = {
        days[0]: {"000001": 100, "000002": 200},
        days[1]: {"000002": 200, "000003": 300},
        days[2]: {"000003": 300, "000004": 400},
        days[3]: {"000004": 400, "000005": 500},
    }

    transitions = compare_adjacent(snapshots, days)

    assert [r["effective_date"] for r in transitions] == [
        "2020-03-25", "2020-03-26", "2020-03-27"]
    assert [r["removed"] for r in transitions] == ["000001", "000002", "000003"]
    assert [r["added"] for r in transitions] == ["000003", "000004", "000005"]


def test_nontrading_day_does_not_break_pair_and_no_change_is_reported():
    before, after = pd.Timestamp("2024-04-30"), pd.Timestamp("2024-05-02")
    snapshots = {before: {"000001": 100}, after: {"000001": 100}}

    transitions = compare_adjacent(snapshots, [before, after])

    assert len(transitions) == 1
    assert transitions[0]["removed_count"] == 0
    assert transitions[0]["added_count"] == 0


def test_cancelled_dated_notice_is_not_expected_change():
    notices = pd.DataFrame([
        {"변경적용일": "2025-03-26", "변경유형": "수시변경", "게시물번호": "1",
         "공지일": "2025-03-20", "제목": "시행", "비고": ""},
        {"변경적용일": "2025-09-09", "변경유형": "수시변경", "게시물번호": "2",
         "공지일": "2025-07-23", "제목": "예정", "비고": "이후 취소 — 실제 미반영"},
        {"변경적용일": None, "변경유형": "수시변경", "게시물번호": "3",
         "공지일": "2025-07-25", "제목": "조건부", "비고": ""},
    ])

    dated, unresolved = classify_notices(notices)

    assert dated == {"2025-03-26"}
    assert set(unresolved["게시물번호"]) == {"2", "3"}


def test_price_check_matches_snapshot_date_and_reports_missing_ticker(tmp_path):
    day = pd.Timestamp("2024-05-02")
    pd.DataFrame({"ISU_CD": ["000001", "08537M"],
                  "TDD_CLSPRC": ["100", "200"]}).to_parquet(tmp_path / "date=20240502.parquet")

    errors = validate_prices({day: {"000001": 101, "000002": 200}}, tmp_path)

    assert len(errors) == 2
    assert errors[0]["ticker"] == "000001" and "종가 불일치" in errors[0]["reason"]
    assert errors[1]["ticker"] == "000002" and errors[1]["reason"] == "Open API에 종목 없음"


def test_experiment_event_check_separates_uncovered_date(tmp_path):
    path = tmp_path / "event_phases.parquet"
    pd.DataFrame([
        {"ticker": "000001", "direction": "편출", "phase": "short_shock", "start": pd.Timestamp("2024-05-02")},
        {"ticker": "000002", "direction": "편입", "phase": "short_shock", "start": pd.Timestamp("2024-05-02")},
        {"ticker": "000003", "direction": "편출", "phase": "short_shock", "start": pd.Timestamp("2020-04-22")},
    ]).to_parquet(path)
    transitions = [{"effective_date": "2024-05-02", "removed_count": 1, "added_count": 1,
                    "removed": "000001", "added": "000002"}]

    rows = compare_experiment_events(transitions, path, "2024-12-30")

    assert [r["status"] for r in rows] == ["unverified_date", "match", "match"]


def test_reconstruct_experiment_membership_from_official_baseline(tmp_path):
    path = tmp_path / "event_phases.parquet"
    pd.DataFrame([
        {"ticker": "000001", "direction": "편출", "phase": "short_shock", "start": pd.Timestamp("2024-05-02")},
        {"ticker": "000002", "direction": "편입", "phase": "short_shock", "start": pd.Timestamp("2024-05-02")},
    ]).to_parquet(path)
    snapshots = {
        pd.Timestamp("2024-04-30"): {"000001": 100},
        pd.Timestamp("2024-05-02"): {"000002": 200},
        pd.Timestamp("2024-05-03"): {"000002": 201, "000003": 300},
    }

    rows = reconstruct_experiment_membership(snapshots, path, "2024-12-30")

    assert rows[0]["baseline"] is True
    assert rows[1]["official_only"] == "" and rows[1]["experiment_only"] == ""
    assert rows[2]["official_only"] == "000003"
