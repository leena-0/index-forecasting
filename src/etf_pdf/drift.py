"""ETF PDF 스냅샷 사이의 일별 주식 비중을 복원한다.

월말(향후 변경 적용일 포함) 스냅샷에서는 PDF 비중으로 재설정하고, 다음
스냅샷 직전까지는 KRX Open API의 종목별 값 변화로 비중을 drift한다.

기본값은 시가총액(``mcap``) 비율이다. Open API 종가는 수정주가가 아니므로
분할·병합 때 가짜 수익률이 생기지만 시가총액은 주식 수 변화를 함께 반영해
그 문제에 상대적으로 강하다. 다만 유상증자 등 회사 전체 주식 수 변화까지
반영하므로 이것 역시 ETF의 정확한 보유 비중이 아닌 대용치다. 비교 실험을
위해 ``close``도 선택할 수 있고 사용 방법은 결과 메타데이터에 기록한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import EtfPdfError

RAW_VALUE_COLUMNS = {"mcap": "MKTCAP", "close": "TDD_CLSPRC"}


def _number(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned.replace({"": pd.NA, "-": pd.NA}), errors="coerce")


def load_openapi_values(
    openapi_dir: str | Path,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    method: str = "mcap",
) -> pd.DataFrame:
    """팀원 수집기의 ``kosdaq_daily/date=YYYYMMDD.parquet``를 공통 형식으로 읽는다."""
    if method not in RAW_VALUE_COLUMNS:
        raise EtfPdfError(f"지원하지 않는 drift 방법: {method}")
    folder = Path(openapi_dir) / "kosdaq_daily"
    if not folder.is_dir():
        raise EtfPdfError(
            f"Open API 종목 폴더가 없음: {folder} (최신 src.krxapi 수집 결과가 필요함)"
        )
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    value_column = RAW_VALUE_COLUMNS[method]
    parts = []
    for path in sorted(folder.glob("date=*.parquet")):
        try:
            date = pd.to_datetime(path.stem[5:], format="%Y%m%d")
        except ValueError as exc:
            raise EtfPdfError(f"Open API 파일명 날짜 오류: {path.name}") from exc
        if date < start or date > end:
            continue
        frame = pd.read_parquet(path)
        if frame.empty:  # 평일이지만 휴장일인 경우 수집기가 0행 파일을 남긴다
            continue
        missing = [c for c in ("ISU_CD", value_column) if c not in frame.columns]
        if missing:
            raise EtfPdfError(f"{path.name}: Open API 필수 열 없음: {missing}")
        ticker = (
            frame["ISU_CD"]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )
        parts.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker": ticker,
                    "value": _number(frame[value_column]),
                }
            )
        )
    if not parts:
        raise EtfPdfError(
            f"{folder}에 {start.date()}~{end.date()} 범위의 거래일 데이터가 없음"
        )
    values = pd.concat(parts, ignore_index=True).sort_values(
        ["date", "ticker"], ignore_index=True
    )
    if values.duplicated(["date", "ticker"]).any():
        raise EtfPdfError("Open API 데이터에 날짜·종목코드 중복이 있음")
    return values


def build_daily_weights(
    snapshots: pd.DataFrame,
    values: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    method: str = "mcap",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """스냅샷 비중과 일별 종목 값을 이용해 일별 비중을 복원한다.

    결과는 주식별 긴 형식과 날짜별 품질 표 두 개다. 구성종목의 값이 하나라도
    없거나 0 이하인 날짜는 일부 종목만으로 재정규화하지 않고 그 날짜 전체를
    ``complete=False`` 및 비중 결측으로 표시한다.
    """
    if method not in RAW_VALUE_COLUMNS:
        raise EtfPdfError(f"지원하지 않는 drift 방법: {method}")
    snapshot_required = {"snapshot_date", "ticker", "equity_weight_pct"}
    value_required = {"date", "ticker", "value"}
    if missing := sorted(snapshot_required - set(snapshots.columns)):
        raise EtfPdfError(f"스냅샷 필수 열 없음: {missing}")
    if missing := sorted(value_required - set(values.columns)):
        raise EtfPdfError(f"일별 값 필수 열 없음: {missing}")

    snapshots = snapshots.loc[:, list(snapshot_required)].copy()
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])
    snapshots["ticker"] = snapshots["ticker"].astype("string").str.zfill(6)
    snapshots["equity_weight_pct"] = pd.to_numeric(
        snapshots["equity_weight_pct"], errors="coerce"
    )
    values = values.loc[:, list(value_required)].copy()
    values["date"] = pd.to_datetime(values["date"])
    values["ticker"] = values["ticker"].astype("string").str.zfill(6)
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    if snapshots.duplicated(["snapshot_date", "ticker"]).any():
        raise EtfPdfError("스냅샷에 날짜·종목코드 중복이 있음")
    if values.duplicated(["date", "ticker"]).any():
        raise EtfPdfError("일별 값에 날짜·종목코드 중복이 있음")
    if (
        snapshots[["snapshot_date", "ticker", "equity_weight_pct"]]
        .isna()
        .any(axis=None)
    ):
        raise EtfPdfError("스냅샷 날짜·종목·비중에 결측이 있음")

    sums = snapshots.groupby("snapshot_date")["equity_weight_pct"].sum()
    bad_sums = sums[~np.isclose(sums, 100.0, atol=1e-6)]
    if not bad_sums.empty:
        raise EtfPdfError(f"스냅샷 정규화 비중합이 100%가 아님: {bad_sums.to_dict()}")
    dates = pd.DatetimeIndex(sorted(values["date"].unique()))
    if dates.empty:
        raise EtfPdfError("일별 값이 비어 있음")
    last_date = pd.Timestamp(as_of) if as_of is not None else dates[-1]
    if last_date > dates[-1]:
        raise EtfPdfError(
            f"as_of {last_date.date()}까지 요청했지만 일별 값은 {dates[-1].date()}까지만 있음"
        )

    anchors = pd.DatetimeIndex(sorted(snapshots["snapshot_date"].unique()))
    rows, quality_rows = [], []
    indexed_values = values.set_index(["date", "ticker"])["value"]
    for index, anchor_date in enumerate(anchors):
        if anchor_date > last_date:
            break
        next_date = anchors[index + 1] if index + 1 < len(anchors) else None
        interval_end = (
            min(last_date, next_date - pd.Timedelta(days=1))
            if next_date is not None
            else last_date
        )
        interval_dates = dates[(dates >= anchor_date) & (dates <= interval_end)]
        if len(interval_dates) == 0 or interval_dates[0] != anchor_date:
            raise EtfPdfError(
                f"스냅샷 기준일 {anchor_date.date()}의 Open API 값이 없음"
            )

        anchor = snapshots[snapshots["snapshot_date"] == anchor_date].set_index(
            "ticker"
        )
        tickers = pd.Index(anchor.index, dtype="string")
        grid = pd.MultiIndex.from_product(
            [interval_dates, tickers], names=["date", "ticker"]
        ).to_frame(index=False)
        grid["anchor_date"] = anchor_date
        grid["next_snapshot_date"] = next_date
        grid["snapshot_weight_pct"] = grid["ticker"].map(anchor["equity_weight_pct"])
        grid["value"] = indexed_values.reindex(
            pd.MultiIndex.from_frame(grid[["date", "ticker"]])
        ).to_numpy()
        anchor_values = indexed_values.reindex(
            pd.MultiIndex.from_product([[anchor_date], tickers])
        )
        invalid_anchor = anchor_values.isna() | (anchor_values <= 0)
        if invalid_anchor.any():
            missing_tickers = tickers[invalid_anchor.to_numpy()].tolist()
            raise EtfPdfError(
                f"스냅샷 기준일 {anchor_date.date()}의 종목 값 결측/0 이하: "
                f"{missing_tickers[:10]}"
            )
        grid["anchor_value"] = grid["ticker"].map(
            pd.Series(anchor_values.to_numpy(), index=tickers)
        )
        invalid = grid["value"].isna() | (grid["value"] <= 0)
        grid["drift_value"] = (
            grid["snapshot_weight_pct"] * grid["value"] / grid["anchor_value"]
        )
        denominator = grid["drift_value"].groupby(grid["date"]).transform("sum")
        missing_count = invalid.groupby(grid["date"]).transform("sum")
        grid["complete"] = missing_count.eq(0)
        grid["weight_pct"] = (grid["drift_value"] / denominator * 100.0).where(
            grid["complete"]
        )
        grid["weight_source"] = np.where(
            grid["date"].eq(anchor_date), "snapshot", f"drift_{method}"
        )
        grid.loc[grid["date"].eq(anchor_date), "weight_pct"] = grid.loc[
            grid["date"].eq(anchor_date), "snapshot_weight_pct"
        ]
        grid["reason"] = np.where(grid["complete"], "", "구성종목 일별 값 결측/0 이하")
        rows.append(
            grid[
                [
                    "date",
                    "ticker",
                    "weight_pct",
                    "weight_source",
                    "anchor_date",
                    "next_snapshot_date",
                    "complete",
                    "reason",
                ]
            ]
        )
        quality = grid.groupby("date", as_index=False).agg(
            anchor_date=("anchor_date", "first"),
            next_snapshot_date=("next_snapshot_date", "first"),
            weight_source=("weight_source", "first"),
            complete=("complete", "first"),
            missing_tickers=("value", lambda x: int((x.isna() | (x <= 0)).sum())),
            weight_sum_pct=("weight_pct", lambda x: x.sum(min_count=1)),
            ticker_count=("ticker", "size"),
        )
        quality["reason"] = np.where(
            quality["complete"], "", "구성종목 일별 값 결측/0 이하"
        )
        quality_rows.append(quality)

    if not rows:
        raise EtfPdfError("as_of 이전에 사용할 수 있는 ETF PDF 스냅샷이 없음")
    daily = pd.concat(rows, ignore_index=True).sort_values(
        ["date", "ticker"], ignore_index=True
    )
    quality = pd.concat(quality_rows, ignore_index=True).sort_values(
        "date", ignore_index=True
    )
    return daily, quality


def prepare_daily_weights(
    snapshots_path: str | Path,
    openapi_dir: str | Path,
    output_dir: str | Path,
    as_of: str = "2025-12-30",
    method: str = "mcap",
) -> dict:
    """표준 월말 스냅샷과 Open API 원본을 결합해 일별 결과를 저장한다."""
    snapshots_path = Path(snapshots_path)
    if not snapshots_path.is_file():
        raise EtfPdfError(
            f"표준 스냅샷 파일이 없음: {snapshots_path} "
            "(먼저 python -m src.etf_pdf prepare 실행)"
        )
    snapshots = pd.read_parquet(snapshots_path)
    start = pd.to_datetime(snapshots["snapshot_date"]).min()
    values = load_openapi_values(openapi_dir, start, as_of, method)
    daily, quality = build_daily_weights(snapshots, values, as_of, method)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_dir / "daily_weights.parquet", index=False)
    quality.to_parquet(output_dir / "daily_weight_quality.parquet", index=False)
    meta = {
        "method": method,
        "method_note": (
            "market-cap ratio proxy; relatively robust to splits but affected by share issuance"
            if method == "mcap"
            else "unadjusted close-price ratio; corporate actions can create false jumps"
        ),
        "role": "ETF-holdings-based proxy; not official KOSDAQ150 index weights",
        "as_of": str(pd.Timestamp(as_of).date()),
        "first_date": str(quality["date"].min().date()),
        "last_date": str(quality["date"].max().date()),
        "dates": len(quality),
        "complete_dates": int(quality["complete"].sum()),
        "incomplete_dates": int((~quality["complete"]).sum()),
        "rows": len(daily),
        "outputs": ["daily_weights.parquet", "daily_weight_quality.parquet"],
    }
    (output_dir / "daily_weights_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
