"""KRX 기준가와 수정주가 수익률 비교 검증 테스트."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.price_validation.__main__ import validate
from src.price_validation.compare import (
    compare_returns,
    load_adjusted,
    load_krx,
    summarize,
)


class PriceValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.api = self.tmp / "openapi/kosdaq_daily"
        self.adjusted = self.tmp / "adjusted"
        self.api.mkdir(parents=True)
        self.adjusted.mkdir()

        def row(date, o, h, l, c, cmp, fluc, shares=100):
            return {"BAS_DD": date, "ISU_CD": "000001", "TDD_OPNPRC": str(o),
                    "TDD_HGPRC": str(h), "TDD_LWPRC": str(l), "TDD_CLSPRC": str(c),
                    "CMPPREVDD_PRC": str(cmp), "FLUC_RT": str(fluc),
                    "ACC_TRDVOL": "1000", "LIST_SHRS": str(shares)}

        raw = {
            "20220103": row("20220103", 99, 101, 98, 100, 0, 0),
            "20220104": row("20220104", 101, 103, 100, 102, 2, 2),
            # 전일 종가 102 → 기업행사 기준가 51. 수정주가는 과거 가격을 1/2로 조정한다.
            "20220105": row("20220105", 52, 54, 51, 53, 2, 3.92, shares=200),
        }
        for date, record in raw.items():
            pd.DataFrame([record]).to_parquet(self.api / f"date={date}.parquet", index=False)
        pd.DataFrame().to_parquet(self.api / "date=20220102.parquet", index=False)  # 휴장일 표시 파일

        pd.DataFrame({"date": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
                      "open": [49.5, 50.5, 52.0], "high": [50.5, 51.5, 54.0],
                      "low": [49.0, 50.0, 51.0], "close": [50.0, 51.0, 53.0],
                      "volume": 1000}).to_csv(self.adjusted / "000001.csv", index=False)

    def compared(self):
        krx = load_krx(self.tmp / "openapi", "2022-01-03", "2022-01-05")
        adjusted = load_adjusted([self.adjusted], "2022-01-03", "2022-01-05")
        return krx, adjusted, compare_returns(krx, adjusted, tolerance=1e-12)

    def test_regular_and_split_returns_match(self):
        krx, _, comparison = self.compared()
        self.assertEqual(len(comparison), 2)  # 첫날은 수익률 계산 불가
        self.assertEqual(comparison["day_type"].tolist(), ["regular", "base_adjustment"])
        self.assertTrue(np.allclose(comparison["ret_abs_error"], 0, atol=1e-15))
        self.assertTrue(np.allclose(comparison["overnight_abs_error"], 0, atol=1e-15))
        split = krx[krx["date"] == pd.Timestamp("2022-01-05")].iloc[0]
        self.assertEqual(split["base_price"], 51)
        self.assertTrue(split["base_adjustment"])
        self.assertTrue(split["corp_action"])

    def test_summary_and_mismatch(self):
        _, adjusted, comparison = self.compared()
        adjusted.loc[adjusted["date"] == pd.Timestamp("2022-01-05"), "adjusted_ret"] += 0.01
        krx = load_krx(self.tmp / "openapi", "2022-01-03", "2022-01-05")
        comparison = compare_returns(krx, adjusted, tolerance=1e-4)
        self.assertEqual(int(comparison["mismatch"].sum()), 1)
        summary = summarize(comparison).set_index(["day_type", "metric"])
        self.assertAlmostEqual(summary.loc[("base_adjustment", "ret"), "max_abs_error"], 0.01)
        self.assertEqual(summary.loc[("regular", "ret"), "n"], 1)

    def test_validate_writes_reproducible_outputs(self):
        out = self.tmp / "out"
        comparison, summary, mismatches, meta = validate(
            self.tmp / "openapi", [self.adjusted], out, "2022-01-03", "2022-01-05", tolerance=1e-12)
        self.assertEqual((len(comparison), len(summary), len(mismatches)), (2, 6, 0))
        self.assertEqual(meta["counts"]["base_adjustment_rows"], 1)
        for name in ["comparison.csv", "summary.csv", "mismatches.csv", "meta.json", "SUMMARY.md"]:
            self.assertTrue((out / name).exists(), name)
        text = (out / "SUMMARY.md").read_text(encoding="utf-8")
        self.assertIn("기준가 조정일 1", text)
        self.assertIn("외부 검증에만 사용", text)


if __name__ == "__main__":
    unittest.main()
