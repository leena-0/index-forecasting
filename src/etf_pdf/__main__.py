"""ETF PDF 원본 검증·표준화.

python -m src.etf_pdf validate
python -m src.etf_pdf prepare
"""

import argparse
import json
import sys
from pathlib import Path

from .drift import prepare_daily_weights
from .loader import EtfPdfError, prepare_directory, validate_directory

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.etf_pdf")
    parser.add_argument("command", choices=["validate", "prepare", "build-daily"])
    parser.add_argument("--input", default=str(REPO / "data" / "manual" / "etf_pdf"))
    parser.add_argument("--output", default=str(REPO / "data" / "interim" / "etf_pdf"))
    parser.add_argument("--start", default="2015-10", help="첫 월 (YYYY-MM)")
    parser.add_argument("--end", default="2025-12", help="마지막 월 (YYYY-MM)")
    parser.add_argument("--fund-code", default="229200")
    parser.add_argument("--openapi", default=str(REPO / "data" / "raw" / "openapi"))
    parser.add_argument("--as-of", default="2025-12-30")
    parser.add_argument("--method", choices=["mcap", "close"], default="mcap")
    parser.add_argument(
        "--weight-tolerance",
        type=float,
        default=0.25,
        help="전체 비중합의 100% 허용오차(%p)",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            equities, cash, snapshots = validate_directory(
                args.input, args.start, args.end, args.fund_code, args.weight_tolerance
            )
            formats = snapshots["source_format"].value_counts().sort_index().to_dict()
            print(
                f"검사 통과: 스냅샷 {len(snapshots)}개, 주식 행 {len(equities)}개, "
                f"현금 행 {len(cash)}개, 형식 {formats}"
            )
        elif args.command == "prepare":
            meta = prepare_directory(
                args.input,
                args.output,
                args.start,
                args.end,
                args.fund_code,
                args.weight_tolerance,
            )
            print(json.dumps(meta, ensure_ascii=False, indent=2))
            print(f"저장: {Path(args.output)}")
        else:
            meta = prepare_daily_weights(
                Path(args.output) / "monthly_equities.parquet",
                args.openapi,
                args.output,
                args.as_of,
                args.method,
            )
            print(json.dumps(meta, ensure_ascii=False, indent=2))
            print(f"저장: {Path(args.output)}")
    except EtfPdfError as exc:
        print(f"검사 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
