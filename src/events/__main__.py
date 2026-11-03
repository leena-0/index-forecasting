"""편입·편출 공지일 데이터 검사와 변환.

    python -m src.events validate [--input data/raw/events/announcements.csv]
    python -m src.events from-notices --notices data/manual/events/공지목록.csv [--source-type crawled]
"""

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

from .loader import EventDataError, load_announcements
from .notices import load_notices, notices_to_announcements

REPO = Path(__file__).resolve().parents[2]


def from_notices(args):
    """공지 목록(공지 단위)과 구성종목 변화 이벤트를 짝지어 공지일 CSV(종목 단위)를 만든다."""
    import yaml
    from dotenv import load_dotenv

    from src.pipeline import legacy

    load_dotenv(REPO / ".env", override=False)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    legacy_dir = cfg.get("legacy_dir") or os.environ.get("LEGACY_DATA_DIR")
    if not legacy_dir:
        print("configs/pipeline.yaml의 legacy_dir 또는 환경변수 LEGACY_DATA_DIR를 설정하세요.", file=sys.stderr)
        return 1
    events = legacy.load(legacy_dir, cfg["as_of"], cfg["placeholder_announce"])["events"]
    notices_path = Path(args.notices)
    collected_at = args.collected_at or dt.datetime.fromtimestamp(notices_path.stat().st_mtime).astimezone() \
        .isoformat(timespec="seconds")
    rows, unmatched, unused = notices_to_announcements(events, load_notices(notices_path), args.source_type,
                                                       collected_at)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False)
    load_announcements(output)  # docs/EVENTS.md 형식 검사
    print(f"공지일 CSV {len(rows)}건 저장: {output} (source_type={args.source_type}, collected_at={collected_at})")
    if len(unmatched):
        by_date = unmatched.groupby(["effective_date", "event_type"])["ticker"].agg(",".join)
        print(f"공지와 짝이 없는 이벤트 {len(unmatched)}건 — 공지 목록에 변경적용일을 채우거나 공지를 추가해야 합니다:")
        for (date, event_type), tickers in by_date.items():
            print(f"  {date.date()} {event_type}: {tickers}")
    if len(unused):
        print(f"짝지어진 이벤트가 없는 공지 {len(unused)}건 (as_of 이후 등): {unused['notice_id'].tolist()}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.events")
    parser.add_argument("command", choices=["validate", "from-notices"])
    parser.add_argument(
        "--input", default=str(REPO / "data" / "raw" / "events" / "announcements.csv")
    )
    parser.add_argument("--as-of", default=None, help="이 날짜 이후 적용 이벤트 제외 (YYYY-MM-DD)")
    parser.add_argument("--notices", help="from-notices: 공지 목록 CSV")
    parser.add_argument("--output", default=str(REPO / "data" / "raw" / "events" / "announcements.csv"),
                        help="from-notices: 저장할 공지일 CSV")
    parser.add_argument("--source-type", choices=["manual", "crawled"], default="crawled",
                        help="from-notices: 공지 목록을 얻은 방식")
    parser.add_argument("--collected-at", default=None, help="from-notices: 수집 일시 (기본: 공지 목록 파일 수정 시각)")
    parser.add_argument("--config", default=str(REPO / "configs" / "pipeline.yaml"),
                        help="from-notices: 구성종목 이벤트를 읽을 설정 (legacy_dir, as_of)")
    args = parser.parse_args(argv)
    try:
        if args.command == "from-notices":
            if not args.notices:
                parser.error("from-notices에는 --notices가 필요합니다.")
            return from_notices(args)
        events = load_announcements(args.input, args.as_of)
    except EventDataError as exc:
        print(f"검사 실패: {exc}", file=sys.stderr)
        return 1
    print(
        f"검사 통과: 이벤트 {len(events)}건, 종목 {events['ticker'].nunique()}개, "
        f"적용일 {events['effective_date'].min().date()}~{events['effective_date'].max().date()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
