"""ETF PDF 수동 원본 어댑터 단위 테스트 (네트워크·실데이터 불필요)."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
