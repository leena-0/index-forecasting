"""KRX 기준가 수익률과 기존 네이버 수정주가 수익률 비교.

python -m src.price_validation --adjusted-dir /path/to/raw/aie_history/stocks /path/to/raw/stocks
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from src.pipeline.__main__ import git_commit

from .compare import compare_returns, load_adjusted, load_krx, summarize

REPO = Path(__file__).resolve().parents[2]


def _markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{x:.8g}" if isinstance(x, float) else str(x) for x in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def validate(openapi_dir, adjusted_dirs, out_dir, start=None, end=None, tickers=None, tolerance=1e-4):
    krx = load_krx(openapi_dir, start, end, tickers)
    adjusted = load_adjusted(adjusted_dirs, start, end, tickers)
    comparison = compare_returns(krx, adjusted, tolerance)
    if comparison.empty:
        raise ValueError("두 데이터에 공통으로 존재하면서 수익률을 계산할 수 있는 행이 없습니다.")
    summary = summarize(comparison)
    mismatches = comparison[comparison["mismatch"]].sort_values(
        ["ret_abs_error", "overnight_abs_error"], ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    mismatches.to_csv(out_dir / "mismatches.csv", index=False)
    period = [comparison["date"].min().date(), comparison["date"].max().date()]
    counts = {"krx_rows": len(krx), "adjusted_rows": len(adjusted), "compared_rows": len(comparison),
              "regular_rows": int((comparison["day_type"] == "regular").sum()),
              "base_adjustment_rows": int((comparison["day_type"] == "base_adjustment").sum()),
              "mismatch_rows": int(comparison["mismatch"].sum())}
    meta = {"created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_commit(), "period": list(map(str, period)), "tolerance": tolerance,
            "openapi_dir": str(openapi_dir), "adjusted_dirs": list(map(str, adjusted_dirs)),
            "tickers": sorted(tickers) if tickers else None, "counts": counts,
            "reportable": False,
            "note": "네이버 수정주가는 외부 검증에만 사용하며 최종 모델 입력이 아님"}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    shown = summary.copy()
    for column in ["mean_abs_error", "median_abs_error", "max_abs_error", "mean_error"]:
        shown[column] = shown[column].map(lambda x: "" if pd.isna(x) else f"{x:.8g}")
    report = ("# KRX 기준가 수익률 외부 검증\n\n"
              f"- 기간: {period[0]} ~ {period[1]}\n"
              f"- 비교 행: {counts['compared_rows']:,} (일반일 {counts['regular_rows']:,}, "
              f"기준가 조정일 {counts['base_adjustment_rows']:,})\n"
              f"- 허용 오차 초과: {counts['mismatch_rows']:,}행 (허용치 {tolerance:g})\n"
              "- 네이버 수정주가는 외부 검증에만 사용하며 모델 입력으로 사용하지 않음\n\n"
              "## 오차 요약\n\n" + _markdown_table(shown) + "\n")
    (out_dir / "SUMMARY.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"저장: {out_dir}")
    return comparison, summary, mismatches, meta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.price_validation")
    ap.add_argument("--openapi-dir", default="data/raw/openapi")
    ap.add_argument("--adjusted-dir", nargs="+", required=True, help="기존 네이버 수정 OHLC CSV 폴더")
    ap.add_argument("--out", default="data/validation/base_price")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--ticker", action="append", dest="tickers", help="비교할 6자리 종목코드(여러 번 지정 가능)")
    ap.add_argument("--tolerance", type=float, default=1e-4, help="로그수익률 허용 오차(기본 약 1bp)")
    args = ap.parse_args(argv)
    validate(REPO / args.openapi_dir, args.adjusted_dir, REPO / args.out,
             args.start, args.end, args.tickers, args.tolerance)


if __name__ == "__main__":
    main()
