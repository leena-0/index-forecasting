"""편입·편출 공지일 데이터 검사.

    python -m src.events validate [--input data/raw/events/announcements.csv]
"""

import argparse
import sys
from pathlib import Path

from .loader import EventDataError, load_announcements

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.events")
    parser.add_argument("command", choices=["validate"])
    parser.add_argument(
        "--input", default=str(REPO / "data" / "raw" / "events" / "announcements.csv")
    )
    parser.add_argument("--as-of", default=None, help="이 날짜 이후 적용 이벤트 제외 (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    try:
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
