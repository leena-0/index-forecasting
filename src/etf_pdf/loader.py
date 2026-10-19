"""KODEX 코스닥150 ETF PDF 원본(XLSX/CSV)을 표준 형식으로 변환한다.

여기서 PDF는 문서 확장자가 아니라 Portfolio Deposit File을 뜻한다. 원본은
연도에 따라 XLSX 또는 CP949 CSV로 제공되므로 둘 다 직접 읽고 원본 형식은
바꾸지 않는다.

월말 PDF는 KOSDAQ150 공식 지수 비중이 아니라 ETF 보유내역의 스냅샷이다.
따라서 출력 열도 ``proxy``임을 명시하고, 현금을 제외한 주식끼리 정규화한
비중과 원본 비중을 함께 보존한다.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "종목코드",
    "구성종목명",
    "주식수(계약수)",
    "평가금액",
    "시가총액",
    "시가총액 구성비중",
)
OUTPUT_COLUMNS = (
    "snapshot_date",
    "ticker",
    "name",
    "units",
    "valuation",
    "market_value",
    "raw_weight_pct",
    "equity_weight_pct",
    "weight_source",
    "source_file",
    "source_sha256",
    "source_format",
)


class EtfPdfError(ValueError):
    """원본 파일 또는 월별 묶음 검증 실패."""


@dataclass(frozen=True)
class Snapshot:
    equities: pd.DataFrame
    cash: pd.DataFrame
    summary: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _snapshot_date(path: Path, fund_code: str) -> pd.Timestamp:
    match = re.fullmatch(
        rf"{re.escape(fund_code)}_(\d{{8}})\.(xlsx|csv)", path.name, re.IGNORECASE
    )
    if not match:
        raise EtfPdfError(
            f"파일명 형식 오류: {path.name} (예: {fund_code}_20251230.csv)"
        )
    try:
        return pd.to_datetime(match.group(1), format="%Y%m%d")
    except ValueError as exc:
        raise EtfPdfError(f"파일명의 날짜가 잘못됨: {path.name}") from exc


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Workbook contains no default style"
            )
            frame = pd.read_excel(path, dtype=str)
    elif path.suffix.lower() == ".csv":
        # KRX에서 내려받은 2022년 이후 파일은 CP949다. utf-8-sig도 허용해
        # 사용자가 다시 저장한 파일을 읽되, 원본 변환을 요구하지는 않는다.
        try:
            frame = pd.read_csv(path, encoding="cp949", dtype=str)
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    else:
        raise EtfPdfError(f"지원하지 않는 확장자: {path.name}")
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise EtfPdfError(f"{path.name}: 필수 열 없음: {missing}")
    return frame.loc[:, REQUIRED_COLUMNS].copy()


def _number(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned.replace({"": pd.NA, "-": pd.NA}), errors="coerce")


def read_snapshot(
    path: str | Path, fund_code: str = "229200", weight_tolerance: float = 0.25
) -> Snapshot:
    """원본 한 개를 읽고 주식·현금 표와 검증 요약을 반환한다.

    ``raw_weight_pct``는 파일의 비중을 그대로 보존한다. ``equity_weight_pct``는
    현금을 제외한 주식의 원본 비중 합으로 다시 나눠 합계가 100이 되게 한다.
    """
    path = Path(path)
    date = _snapshot_date(path, fund_code)
    raw = _read_raw(path)

    code = (
        raw["종목코드"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    name = raw["구성종목명"].astype("string").str.strip()
    is_cash = code.str.startswith("KRD", na=False) | name.str.contains(
        r"^원화(?:예금|현금)$", na=False
    )
    numeric_code = code.str.fullmatch(r"\d+", na=False)
    ticker = code.mask(numeric_code, code.str.zfill(6))

    normalized = pd.DataFrame(
        {
            "snapshot_date": date,
            "ticker": ticker,
            "name": name,
            "units": _number(raw["주식수(계약수)"]),
            "valuation": _number(raw["평가금액"]),
            "market_value": _number(raw["시가총액"]),
            "raw_weight_pct": _number(raw["시가총액 구성비중"]),
        }
    )
    normalized["source_file"] = path.name
    normalized["source_sha256"] = _sha256(path)
    normalized["source_format"] = path.suffix.lower().lstrip(".")

    equities = normalized.loc[~is_cash].copy()
    cash = normalized.loc[is_cash].copy()
    bad_codes = equities.loc[
        ~equities["ticker"].str.fullmatch(r"\d{6}", na=False), "ticker"
    ].tolist()
    if bad_codes:
        raise EtfPdfError(
            f"{path.name}: 6자리 주식 종목코드가 아닌 값: {bad_codes[:5]}"
        )
    if equities["ticker"].duplicated().any():
        dupes = (
            equities.loc[equities["ticker"].duplicated(False), "ticker"]
            .unique()
            .tolist()
        )
        raise EtfPdfError(f"{path.name}: 종목코드 중복: {dupes[:5]}")
    if len(cash) != 1:
        raise EtfPdfError(
            f"{path.name}: 원화 현금/예금 행이 1개여야 하나 {len(cash)}개"
        )
    if equities.empty:
        raise EtfPdfError(f"{path.name}: 주식 구성종목이 없음")
    required_values = equities[["name", "market_value", "raw_weight_pct"]]
    if required_values.isna().any(axis=None):
        raise EtfPdfError(f"{path.name}: 주식의 종목명·시가총액·비중에 결측이 있음")
    if (equities["raw_weight_pct"] < 0).any():
        raise EtfPdfError(f"{path.name}: 주식 비중에 음수가 있음")

    total_weight = float(normalized["raw_weight_pct"].sum())
    if abs(total_weight - 100.0) > weight_tolerance:
        raise EtfPdfError(
            f"{path.name}: 전체 비중 합 {total_weight:.4f}%가 100%에서 "
            f"허용오차 {weight_tolerance:.4f}%보다 크게 벗어남"
        )
    equity_total = float(equities["raw_weight_pct"].sum())
    if equity_total <= 0:
        raise EtfPdfError(f"{path.name}: 주식 비중 합이 0 이하")
    equities["equity_weight_pct"] = equities["raw_weight_pct"] / equity_total * 100.0
    equities["weight_source"] = "etf_pdf_month_end_proxy"
    cash["equity_weight_pct"] = pd.NA
    cash["weight_source"] = "etf_pdf_cash"

    equities = equities.loc[:, OUTPUT_COLUMNS].sort_values("ticker", ignore_index=True)
    cash = cash.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)
    summary = {
        "snapshot_date": date,
        "source_file": path.name,
        "source_sha256": normalized["source_sha256"].iat[0],
        "source_format": normalized["source_format"].iat[0],
        "equity_rows": len(equities),
        "cash_rows": len(cash),
        "total_weight_pct": total_weight,
        "equity_raw_weight_pct": equity_total,
        "cash_raw_weight_pct": float(cash["raw_weight_pct"].sum()),
    }
    return Snapshot(equities=equities, cash=cash, summary=summary)


def _expected_periods(start: str, end: str) -> pd.PeriodIndex:
    try:
        periods = pd.period_range(
            pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M"
        )
    except ValueError as exc:
        raise EtfPdfError(f"월 범위 오류: {start} ~ {end} (YYYY-MM)") from exc
    if periods.empty:
        raise EtfPdfError(f"월 범위가 비어 있음: {start} ~ {end}")
    return periods


def validate_directory(
    input_dir: str | Path,
    start: str = "2015-10",
    end: str = "2025-12",
    fund_code: str = "229200",
    weight_tolerance: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """폴더 전체의 파일·월 연속성·내용을 검증하고 결합된 표를 반환한다."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise EtfPdfError(f"입력 폴더가 없음: {input_dir}")
    files = sorted(
        [*input_dir.glob(f"{fund_code}_*.xlsx"), *input_dir.glob(f"{fund_code}_*.csv")]
    )
    if not files:
        raise EtfPdfError(f"{input_dir}에 {fund_code}_YYYYMMDD.xlsx/csv 파일이 없음")

    snapshots = [read_snapshot(p, fund_code, weight_tolerance) for p in files]
    summaries = pd.DataFrame([s.summary for s in snapshots]).sort_values(
        "snapshot_date", ignore_index=True
    )
    summaries["month"] = summaries["snapshot_date"].dt.to_period("M").astype(str)
    duplicate_months = (
        summaries.loc[summaries["month"].duplicated(False), "month"].unique().tolist()
    )
    if duplicate_months:
        raise EtfPdfError(f"같은 월의 스냅샷이 둘 이상 있음: {duplicate_months}")

    expected = _expected_periods(start, end)
    actual = pd.PeriodIndex(summaries["month"], freq="M")
    missing = expected.difference(actual).astype(str).tolist()
    extra = actual.difference(expected).astype(str).tolist()
    if missing or extra:
        raise EtfPdfError(f"월 범위 불일치: 누락={missing}, 범위 밖={extra}")

    equities = pd.concat(
        [s.equities for s in snapshots], ignore_index=True
    ).sort_values(["snapshot_date", "ticker"], ignore_index=True)
    cash = pd.concat([s.cash for s in snapshots], ignore_index=True).sort_values(
        "snapshot_date", ignore_index=True
    )
    return equities, cash, summaries


def prepare_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    start: str = "2015-10",
    end: str = "2025-12",
    fund_code: str = "229200",
    weight_tolerance: float = 0.25,
) -> dict:
    """검증을 통과한 월말 PDF를 Parquet와 메타데이터로 저장한다."""
    equities, cash, summaries = validate_directory(
        input_dir, start, end, fund_code, weight_tolerance
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equities.to_parquet(output_dir / "monthly_equities.parquet", index=False)
    cash.to_parquet(output_dir / "monthly_cash.parquet", index=False)
    summaries.to_parquet(output_dir / "snapshots.parquet", index=False)

    meta = {
        "fund_code": fund_code,
        "fund_name": "KODEX 코스닥150",
        "source": "ETF PDF (Portfolio Deposit File), manual download",
        "role": "KOSDAQ150 index-weight proxy; not official index weights",
        "start_month": start,
        "end_month": end,
        "snapshot_count": len(summaries),
        "equity_rows": len(equities),
        "cash_rows": len(cash),
        "formats": summaries["source_format"].value_counts().sort_index().to_dict(),
        "weight_tolerance_pct": weight_tolerance,
        "normalization": "cash excluded; equities rescaled to 100% per snapshot",
        "outputs": [
            "monthly_equities.parquet",
            "monthly_cash.parquet",
            "snapshots.parquet",
        ],
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
