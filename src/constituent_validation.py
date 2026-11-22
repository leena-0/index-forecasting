"""수동 다운로드한 KOSDAQ150 구성종목 스냅샷을 KRX 일별 데이터·공지 목록과 대조한다.

엑셀 원본과 출력은 data/ 아래에 두어 Git에 올리지 않는다. 이 모듈은 기존 구성종목
이력을 변경하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import warnings
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


SNAPSHOT_NAME = re.compile(r"kosdaq150_(\d{4}-\d{2}-\d{2})\.xlsx")
REQUIRED_COLUMNS = {"종목코드", "종목명", "종가"}


def _ticker(value: object) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"\d{1,6}", text):
        raise ValueError(f"유효하지 않은 종목코드: {text!r}")
    return text.zfill(6)


def _close(value: object) -> int:
    text = str(value).replace(",", "").strip()
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(f"유효하지 않은 종가: {text!r}") from exc
    if number < 0:
        raise ValueError(f"음수 종가: {text!r}")
    return number


def load_snapshots(snapshot_dir: Path) -> tuple[dict[pd.Timestamp, dict[str, int]], list[dict]]:
    files = sorted(snapshot_dir.glob("kosdaq150_*.xlsx"))
    if not files:
        raise ValueError(f"구성종목 파일이 없습니다: {snapshot_dir}")

    snapshots: dict[pd.Timestamp, dict[str, int]] = {}
    manifest = []
    for path in files:
        match = SNAPSHOT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"날짜 형식에 맞지 않는 파일명: {path.name}")
        date = pd.Timestamp(match.group(1))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Workbook contains no default style")
            frame = pd.read_excel(path, dtype=str)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name}: 필수 열 누락 {sorted(missing)}")
        if frame[["종목코드", "종목명", "종가"]].isna().any().any():
            raise ValueError(f"{path.name}: 종목코드·종목명·종가에 빈 값이 있습니다")

        codes = [_ticker(v) for v in frame["종목코드"]]
        if len(codes) != len(set(codes)):
            raise ValueError(f"{path.name}: 종목코드가 중복됩니다")
        if len(codes) > 150 or len(codes) == 0:
            raise ValueError(f"{path.name}: 구성종목 수가 {len(codes)}개입니다")
        closes = [_close(v) for v in frame["종가"]]
        if date in snapshots:
            raise ValueError(f"중복 날짜: {date.date()}")
        snapshots[date] = dict(zip(codes, closes))
        manifest.append({
            "date": date.date().isoformat(),
            "file": path.name,
            "constituents": len(codes),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return snapshots, manifest


def load_calendar(index_dir: Path) -> list[pd.Timestamp]:
    days = []
    for path in index_dir.glob("date=*.parquet"):
        # 휴장일의 빈 parquet도 저장되어 있으므로 파일명만으로 거래일을 정하지 않는다.
        if pq.ParquetFile(path).metadata.num_rows:
            days.append(pd.Timestamp(path.stem.removeprefix("date=")))
    if not days:
        raise ValueError(f"거래일 데이터가 없습니다: {index_dir}")
    return sorted(days)


def validate_prices(snapshots: dict, daily_dir: Path) -> list[dict]:
    errors = []
    for date, stocks in sorted(snapshots.items()):
        path = daily_dir / f"date={date:%Y%m%d}.parquet"
        if not path.exists():
            errors.append({"date": str(date.date()), "ticker": "", "reason": "일별매매정보 파일 없음"})
            continue
        frame = pd.read_parquet(path)
        if not {"ISU_CD", "TDD_CLSPRC"} <= set(frame.columns):
            errors.append({"date": str(date.date()), "ticker": "", "reason": "종목코드·종가 열 없음"})
            continue
        daily = {_ticker(t): _close(c) for t, c in zip(frame["ISU_CD"], frame["TDD_CLSPRC"])
                 if re.fullmatch(r"\d{1,6}", str(t).strip()) and str(c).strip() not in ("", "nan")}
        for ticker, close in stocks.items():
            if ticker not in daily:
                errors.append({"date": str(date.date()), "ticker": ticker, "reason": "Open API에 종목 없음"})
            elif daily[ticker] != close:
                errors.append({"date": str(date.date()), "ticker": ticker,
                               "reason": f"종가 불일치: 엑셀 {close}, Open API {daily[ticker]}"})
    return errors


def compare_adjacent(snapshots: dict, calendar: list[pd.Timestamp]) -> list[dict]:
    next_day = dict(zip(calendar, calendar[1:]))
    rows = []
    for date in sorted(snapshots):
        following = next_day.get(date)
        if following not in snapshots:
            continue
        before, after = set(snapshots[date]), set(snapshots[following])
        removed, added = sorted(before - after), sorted(after - before)
        rows.append({
            "previous_date": str(date.date()), "effective_date": str(following.date()),
            "previous_count": len(before), "current_count": len(after),
            "removed_count": len(removed), "added_count": len(added),
            "removed": ",".join(removed), "added": ",".join(added),
        })
    return rows


def classify_notices(notices: pd.DataFrame) -> tuple[set[str], pd.DataFrame]:
    required = {"변경적용일", "변경유형", "게시물번호", "공지일", "제목", "비고"}
    missing = required - set(notices.columns)
    if missing:
        raise ValueError(f"공지 CSV 필수 열 누락: {sorted(missing)}")
    cancelled = notices["변경유형"].fillna("").str.contains("취소|미실시|일정변경") | \
        notices["비고"].fillna("").str.contains("취소|미반영")
    actual = notices.loc[notices["변경적용일"].notna() & ~cancelled, "변경적용일"]
    return set(actual.astype(str)), notices.loc[notices["변경적용일"].isna() | cancelled].copy()


def compare_legacy(snapshots: dict, path: Path) -> list[dict]:
    frame = pd.read_csv(path, dtype=str)
    if not {"date", "ticker"} <= set(frame.columns):
        raise ValueError("기존 이력에 date·ticker 열이 없습니다")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame["ticker"] = frame["ticker"].map(_ticker)
    old = frame.groupby("date")["ticker"].agg(set).to_dict()
    rows = []
    for date, stocks in sorted(snapshots.items()):
        official = set(stocks)
        existing = old.get(date)
        rows.append({
            "date": str(date.date()), "legacy_available": existing is not None,
            "official_only": ",".join(sorted(official - existing)) if existing is not None else "",
            "legacy_only": ",".join(sorted(existing - official)) if existing is not None else "",
        })
    return rows


def compare_experiment_events(transitions: list[dict], path: Path, as_of: str) -> list[dict]:
    """실험 산출물 short_shock의 시작일(설정상 적용일)을 공식 변화와 대조한다."""
    frame = pd.read_parquet(path)
    required = {"ticker", "direction", "phase", "start"}
    if not required <= set(frame.columns):
        raise ValueError(f"실험 산출물 필수 열 누락: {sorted(required - set(frame.columns))}")
    shocks = frame.loc[frame["phase"] == "short_shock"]
    existing = {(str(row.start)[:10], _ticker(row.ticker), row.direction)
                for row in shocks.itertuples() if str(row.start)[:10] <= as_of}
    official = set()
    covered_dates = set()
    for row in transitions:
        date = row["effective_date"]
        if date > as_of or not (row["removed_count"] or row["added_count"]):
            continue
        covered_dates.add(date)
        official.update((date, ticker, "편출") for ticker in row["removed"].split(",") if ticker)
        official.update((date, ticker, "편입") for ticker in row["added"].split(",") if ticker)
    rows = []
    for date, ticker, direction in sorted(official | existing):
        key = (date, ticker, direction)
        if key in official and key in existing:
            status = "match"
        elif key in official:
            status = "official_only"
        elif date in covered_dates:
            status = "experiment_only"
        else:
            status = "unverified_date"
        rows.append({"effective_date": date, "ticker": ticker, "direction": direction, "status": status})
    return rows


def reconstruct_experiment_membership(snapshots: dict, path: Path, as_of: str) -> list[dict]:
    """첫 공식 스냅샷을 시작 집합으로 실험 이벤트를 순서대로 적용한다."""
    frame = pd.read_parquet(path)
    if not {"ticker", "direction", "phase", "start"} <= set(frame.columns):
        raise ValueError("실험 산출물에 ticker·direction·phase·start 열이 필요합니다")
    shocks = frame.loc[frame["phase"] == "short_shock", ["ticker", "direction", "start"]].copy()
    shocks["date"] = pd.to_datetime(shocks["start"])
    shocks = shocks.sort_values("date")
    first = min(snapshots)
    state = set(snapshots[first])
    events = [row for row in shocks.itertuples() if first < row.date <= pd.Timestamp(as_of)]
    rows = []
    pos = 0
    for date, official_prices in sorted(snapshots.items()):
        if date > pd.Timestamp(as_of):
            break
        while pos < len(events) and events[pos].date <= date:
            event = events[pos]
            ticker = _ticker(event.ticker)
            if event.direction == "편출":
                if ticker not in state:
                    raise ValueError(f"실험 이벤트가 이미 빠진 종목을 편출: {event.date.date()} {ticker}")
                state.remove(ticker)
            elif event.direction == "편입":
                if ticker in state:
                    raise ValueError(f"실험 이벤트가 기존 종목을 다시 편입: {event.date.date()} {ticker}")
                state.add(ticker)
            else:
                raise ValueError(f"알 수 없는 방향: {event.direction}")
            pos += 1
        official = set(official_prices)
        rows.append({"date": str(date.date()), "baseline": date == first,
                     "official_count": len(official), "reconstructed_count": len(state),
                     "official_only": ",".join(sorted(official - state)),
                     "experiment_only": ",".join(sorted(state - official))})
    return rows


def run(snapshot_dir: Path, openapi_dir: Path, notices_path: Path, output_dir: Path,
        legacy_snapshots: Path | None = None, experiment_events: Path | None = None,
        as_of: str = "2024-12-30") -> dict:
    snapshots, manifest = load_snapshots(snapshot_dir)
    calendar = load_calendar(openapi_dir / "kosdaq_index")
    missing_calendar = sorted(set(snapshots) - set(calendar))
    if missing_calendar:
        raise ValueError(f"거래일 달력에 없는 파일 날짜: {[str(x.date()) for x in missing_calendar]}")
    price_errors = validate_prices(snapshots, openapi_dir / "kosdaq_daily")
    transitions = compare_adjacent(snapshots, calendar)
    notices = pd.read_csv(notices_path, encoding="utf-8-sig", dtype=str)
    dated, unresolved = classify_notices(notices)
    changed = {r["effective_date"] for r in transitions if r["removed_count"] or r["added_count"]}
    notice_missing = sorted(changed - dated)
    snapshot_missing = sorted(dated - changed)
    legacy_rows = compare_legacy(snapshots, legacy_snapshots) if legacy_snapshots else None
    experiment_rows = compare_experiment_events(transitions, experiment_events, as_of) if experiment_events else None
    reconstructed = reconstruct_experiment_membership(snapshots, experiment_events, as_of) if experiment_events else None

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
    pd.DataFrame(transitions).to_csv(output_dir / "transitions.csv", index=False)
    pd.DataFrame(price_errors, columns=["date", "ticker", "reason"]).to_csv(output_dir / "price_errors.csv", index=False)
    unresolved.to_csv(output_dir / "unresolved_notices.csv", index=False)
    if legacy_rows is not None:
        pd.DataFrame(legacy_rows).to_csv(output_dir / "legacy_comparison.csv", index=False)
    if experiment_rows is not None:
        pd.DataFrame(experiment_rows).to_csv(output_dir / "experiment_event_comparison.csv", index=False)
        pd.DataFrame(reconstructed).to_csv(output_dir / "experiment_membership_comparison.csv", index=False)

    report = {
        "snapshots": len(snapshots), "rows": sum(map(len, snapshots.values())),
        "first_snapshot": str(min(snapshots).date()),
        "last_snapshot": str(max(snapshots).date()),
        "non_150": [f"{d.date()} ({len(values)})" for d, values in sorted(snapshots.items()) if len(values) != 150],
        "price_errors": len(price_errors), "adjacent_pairs": len(transitions),
        "changes": len(changed), "no_change_pairs": len(transitions) - len(changed),
        "no_change_dates": [r["effective_date"] for r in transitions if not r["removed_count"] and not r["added_count"]],
        "dated_notices": len(dated), "notices_without_effective_date": int(notices["변경적용일"].isna().sum()),
        "changes_without_dated_notice": notice_missing,
        "dated_notices_without_change": snapshot_missing,
        "legacy_compared": legacy_rows is not None,
        "legacy_mismatches": sum(bool(r["official_only"] or r["legacy_only"]) for r in legacy_rows) if legacy_rows else None,
        "experiment_event_status": pd.Series([r["status"] for r in experiment_rows]).value_counts().to_dict()
        if experiment_rows is not None else None,
        "experiment_snapshots": len(reconstructed) if reconstructed is not None else None,
        "experiment_snapshot_mismatches": sum(bool(r["official_only"] or r["experiment_only"])
                                              for r in reconstructed) if reconstructed is not None else None,
    }
    lines = ["# KOSDAQ150 구성종목 파일 검증", "",
             f"- 공식 스냅샷: {report['snapshots']}개, 종목 행 {report['rows']:,}개",
             f"- 파일 날짜 범위: {report['first_snapshot']} ~ {report['last_snapshot']}",
             f"- 같은 날짜 KRX Open API 종가 대조 오류: {report['price_errors']}개",
             f"- 연속 거래일 비교: {report['adjacent_pairs']}쌍, 구성종목 변화 {report['changes']}일, 변화 없음 {report['no_change_pairs']}일",
             f"- 150종목이 아닌 파일: {', '.join(report['non_150']) or '없음'}",
             f"- 연속 거래일 사이 변화 없는 날: {', '.join(report['no_change_dates']) or '없음'}",
             f"- 적용일 있는 유효 공지: {report['dated_notices']}개; 적용일 빈 공지: {report['notices_without_effective_date']}개",
             f"- 공지 적용일이 없는 구성종목 변화: {', '.join(notice_missing) or '없음'}",
             f"- 구성종목 변화와 맞지 않는 공지 적용일: {', '.join(snapshot_missing) or '없음'}",
             f"- 기존 일별 구성종목 이력 직접 대조: {'수행' if legacy_rows is not None else '미수행 (snapshots_2203.csv 없음)'}",
             f"- 실험 이벤트 대조 ({as_of}까지): 일치 {report['experiment_event_status'].get('match', 0)}건, "
             f"스냅샷 없는 날짜 {report['experiment_event_status'].get('unverified_date', 0)}건, "
             f"불일치 {report['experiment_event_status'].get('official_only', 0) + report['experiment_event_status'].get('experiment_only', 0)}건"
             if experiment_rows is not None else "- 실험 이벤트 대조: 미수행",
             f"- 실험 이벤트로 재구성한 스냅샷: {report['experiment_snapshots']}개 중 불일치 "
             f"{report['experiment_snapshot_mismatches']}개" if reconstructed is not None else "- 실험 스냅샷 재구성: 미수행",
             "", "## 해석상 한계", "",
             "- 엑셀 파일의 날짜는 같은 날짜 KRX Open API 종가와 전 행 일치하는 것으로 확인한다.",
             "- 공지 목록은 종목코드를 포함하지 않아 적용일 일치만 검증할 수 있다.",
             "- 스냅샷이 없는 거래일의 구성종목 변화는 이 방법으로 발견할 수 없다.",
             "- 스냅샷 사이의 일별 이력은 직접 대조하지 못했으므로 전체 거래일 이력으로 확정하지 않는다.", ""]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=Path("data/manual/krx_constituents"))
    parser.add_argument("--openapi", type=Path, default=Path("data/raw/openapi"))
    parser.add_argument("--notices", type=Path, default=Path("data/manual/events/kosdaq150_changes_2015_2025.csv"))
    parser.add_argument("--legacy-snapshots", type=Path)
    parser.add_argument("--experiment-events", type=Path, help="실험 산출물 event_phases.parquet")
    parser.add_argument("--as-of", default="2024-12-30")
    parser.add_argument("--output", type=Path, default=Path("data/validation/constituents"))
    args = parser.parse_args()
    result = run(args.snapshots, args.openapi, args.notices, args.output,
                 args.legacy_snapshots, args.experiment_events, args.as_of)
    print((args.output / "SUMMARY.md").read_text(encoding="utf-8"))
    if result["price_errors"]:
        raise SystemExit("종가 대조 오류가 있습니다. price_errors.csv를 확인하세요")


if __name__ == "__main__":
    main()
