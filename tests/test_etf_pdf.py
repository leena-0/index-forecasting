"""ETF PDF 수동 원본 어댑터 단위 테스트 (네트워크·실데이터 불필요)."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.etf_pdf.drift import build_daily_weights, load_openapi_values
from src.etf_pdf.loader import (
    EtfPdfError,
    prepare_directory,
    read_snapshot,
    validate_directory,
)


def sample_frame(cash_weight="1.0"):
    return pd.DataFrame(
        {
            "종목코드": ["35720", "066970", "KRD010010001"],
            "구성종목명": ["카카오", "엘앤에프", "원화예금"],
            "주식수(계약수)": ["10", "20", "-"],
            "평가금액": ["1,000", "2,000", "-"],
            "시가총액": ["1,000", "2,000", "30"],
            "시가총액 구성비중": ["33.0", "66.0", cash_weight],
        }
    )


class EtfPdfTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def write_csv(self, name, frame=None, encoding="cp949"):
        path = self.tmp / name
        (frame if frame is not None else sample_frame()).to_csv(
            path, index=False, encoding=encoding
        )
        return path

    def write_xlsx(self, name, frame=None):
        path = self.tmp / name
        (frame if frame is not None else sample_frame()).to_excel(path, index=False)
        return path

    def test_reads_cp949_and_normalizes_codes_and_weights(self):
        snap = read_snapshot(self.write_csv("229200_20220131.csv"))
        self.assertEqual(snap.equities["ticker"].tolist(), ["035720", "066970"])
        self.assertEqual(snap.cash["ticker"].tolist(), ["KRD010010001"])
        self.assertAlmostEqual(snap.equities["equity_weight_pct"].sum(), 100.0)
        self.assertEqual(snap.summary["total_weight_pct"], 100.0)

    def test_reads_xlsx_without_losing_leading_zero(self):
        snap = read_snapshot(self.write_xlsx("229200_20211230.xlsx"))
        self.assertIn("035720", snap.equities["ticker"].tolist())
        self.assertEqual(snap.summary["source_format"], "xlsx")

    def test_rejects_bad_weight_sum_and_duplicate_ticker(self):
        with self.assertRaisesRegex(EtfPdfError, "비중 합"):
            read_snapshot(self.write_csv("229200_20220131.csv", sample_frame("3.0")))
        duplicate = sample_frame()
        duplicate.loc[1, "종목코드"] = "35720"
        with self.assertRaisesRegex(EtfPdfError, "중복"):
            read_snapshot(self.write_csv("229200_20220228.csv", duplicate))

    def test_directory_requires_every_month_once(self):
        self.write_xlsx("229200_20220131.xlsx")
        self.write_csv("229200_20220228.csv")
        equities, cash, summaries = validate_directory(self.tmp, "2022-01", "2022-02")
        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(cash), 2)
        self.assertEqual(len(equities), 4)
        self.write_csv("229200_20220225.csv")
        with self.assertRaisesRegex(EtfPdfError, "같은 월"):
            validate_directory(self.tmp, "2022-01", "2022-02")

    def test_directory_reports_missing_month(self):
        self.write_csv("229200_20220131.csv")
        with self.assertRaisesRegex(EtfPdfError, "2022-02"):
            validate_directory(self.tmp, "2022-01", "2022-02")

    def test_prepare_writes_reproducible_outputs(self):
        self.write_csv("229200_20220131.csv")
        out = self.tmp / "out"
        meta = prepare_directory(self.tmp, out, "2022-01", "2022-01")
        self.assertEqual(meta["snapshot_count"], 1)
        self.assertTrue((out / "monthly_equities.parquet").exists())
        self.assertTrue((out / "monthly_cash.parquet").exists())
        self.assertTrue((out / "snapshots.parquet").exists())
        saved = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        self.assertIn("not official index weights", saved["role"])


class DailyWeightTest(unittest.TestCase):
    def snapshots(self):
        return pd.DataFrame(
            {
                "snapshot_date": pd.to_datetime(
                    ["2022-01-03", "2022-01-03", "2022-01-05", "2022-01-05"]
                ),
                "ticker": ["000001", "000002", "000001", "000002"],
                "equity_weight_pct": [60.0, 40.0, 50.0, 50.0],
            }
        )

    def values(self):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2022-01-03",
                        "2022-01-03",
                        "2022-01-04",
                        "2022-01-04",
                        "2022-01-05",
                        "2022-01-05",
                        "2022-01-06",
                        "2022-01-06",
                    ]
                ),
                "ticker": ["000001", "000002"] * 4,
                "value": [100, 100, 110, 90, 200, 50, 180, 60],
            }
        )

    def test_snapshot_reset_and_between_snapshot_drift(self):
        daily, quality = build_daily_weights(
            self.snapshots(), self.values(), "2022-01-06", "mcap"
        )
        by_key = daily.set_index(["date", "ticker"])
        self.assertEqual(
            by_key.loc[(pd.Timestamp("2022-01-03"), "000001"), "weight_pct"], 60.0
        )
        expected = 60 * 1.1 / (60 * 1.1 + 40 * 0.9) * 100
        self.assertAlmostEqual(
            by_key.loc[(pd.Timestamp("2022-01-04"), "000001"), "weight_pct"],
            expected,
        )
        self.assertEqual(
            by_key.loc[(pd.Timestamp("2022-01-05"), "000001"), "weight_pct"], 50.0
        )
        self.assertEqual(
            by_key.loc[(pd.Timestamp("2022-01-05"), "000001"), "weight_source"],
            "snapshot",
        )
        self.assertTrue(quality["complete"].all())
        self.assertTrue((quality["weight_sum_pct"].round(10) == 100.0).all())

    def test_missing_constituent_invalidates_whole_day(self):
        values = self.values()
        values.loc[
            (values["date"] == "2022-01-04") & (values["ticker"] == "000002"),
            "value",
        ] = pd.NA
        daily, quality = build_daily_weights(
            self.snapshots(), values, "2022-01-06", "mcap"
        )
        bad = daily[daily["date"] == "2022-01-04"]
        self.assertTrue(bad["weight_pct"].isna().all())
        row = quality.set_index("date").loc[pd.Timestamp("2022-01-04")]
        self.assertFalse(row["complete"])
        self.assertEqual(row["missing_tickers"], 1)

    def test_missing_anchor_value_raises(self):
        values = self.values()
        values.loc[
            (values["date"] == "2022-01-03") & (values["ticker"] == "000002"),
            "value",
        ] = pd.NA
        with self.assertRaisesRegex(EtfPdfError, "기준일"):
            build_daily_weights(self.snapshots(), values, "2022-01-06", "mcap")

    def test_loads_team_openapi_layout(self):
        temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp)
        folder = temp / "kosdaq_daily"
        folder.mkdir()
        pd.DataFrame(
            {
                "ISU_CD": ["1", "000002"],
                "MKTCAP": ["1,000", "2,000"],
                "TDD_CLSPRC": ["10", "20"],
            }
        ).to_parquet(folder / "date=20220103.parquet", index=False)
        pd.DataFrame().to_parquet(folder / "date=20220104.parquet", index=False)
        values = load_openapi_values(temp, "2022-01-03", "2022-01-04", "mcap")
        self.assertEqual(values["ticker"].tolist(), ["000001", "000002"])
        self.assertEqual(values["value"].tolist(), [1000, 2000])


if __name__ == "__main__":
    unittest.main()
