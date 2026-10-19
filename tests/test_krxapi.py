"""src.krxapi(수집)와 src.pipeline.openapi(어댑터) 단위 테스트 — 가짜 클라이언트·가짜 Session, 네트워크 없음."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.krxapi import collect
from src.krxapi.client import ApiError, DailyLimit, OpenApiClient
from src.krxapi.store import SnapshotMismatch, Store
from src.pipeline import openapi
from src.pipeline.target import forward_yz
from tests.test_pipeline import make_legacy

CFG = {"as_of": "2015-01-09", "start": "2015-01-01"}
HOLIDAYS = {"20150101"}


class FakeApi:
    """endpoint별로 정해 둔 응답. 휴장일 지수는 빈 응답."""

    def __init__(self, empty_daily=(), raise_on=None):
        self.calls = 0
        self.log = []
        self.empty_daily = set(empty_daily)
        self.raise_on = raise_on

    def get(self, endpoint, bas_dd):
        self.calls += 1
        self.log.append((endpoint, bas_dd))
        if self.raise_on and self.raise_on[0] == endpoint and bas_dd >= self.raise_on[1]:
            raise self.raise_on[2]
        if endpoint == collect.INDEX_EP:
            if bas_dd in HOLIDAYS:
                return []
            return [{"BAS_DD": bas_dd, "IDX_NM": "코스닥", "CLSPRC_IDX": "600"},
                    {"BAS_DD": bas_dd, "IDX_NM": "코스닥 150", "CLSPRC_IDX": "1,000.5"}]
        if bas_dd in self.empty_daily:
            return []
        return [{"BAS_DD": bas_dd, "ISU_CD": "000001", "TDD_CLSPRC": "100"}]


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.store = Store(self.tmp, "tester")

    def run_c(self, api, limit=None, cfg=CFG):
        return collect.Collector(api, self.store, cfg, log=lambda *_: None).run(limit=limit)

    def files(self, kind):
        return sorted(p.stem for p in (self.tmp / "raw/openapi" / kind).glob("*.parquet"))

    def test_holidays_and_trading_days(self):
        api = FakeApi()
        trading = self.run_c(api)
        weekdays = ["20150101", "20150102", "20150105", "20150106", "20150107", "20150108", "20150109"]
        self.assertEqual(self.files("kosdaq_index"), [f"date={d}" for d in weekdays])
        self.assertEqual(trading, weekdays[1:])
        self.assertEqual(self.files("kosdaq_daily"), [f"date={d}" for d in weekdays[1:]])
        self.assertNotIn((collect.DAILY_EP, "20150101"), api.log)          # 휴장일 종목 요청 안 함
        self.assertEqual(len(self.store.read("raw/openapi/kosdaq_index/date=20150101.parquet")), 0)
        self.assertTrue(all(d <= "20150109" for _, d in api.log))            # as_of 이후 요청 안 함

    def test_rerun_skips(self):
        self.run_c(FakeApi())
        api = FakeApi()
        self.run_c(api)
        self.assertEqual(api.calls, 0)

    def test_empty_trading_day_is_failure_and_retried(self):
        self.run_c(FakeApi(empty_daily={"20150106"}))
        self.assertNotIn("date=20150106", self.files("kosdaq_daily"))
        api = FakeApi()
        self.run_c(api)
        self.assertEqual(api.log, [(collect.DAILY_EP, "20150106")])

    def test_daily_limit_stops_and_writes_manifest(self):
        with self.assertRaises(DailyLimit):
            self.run_c(FakeApi(raise_on=(collect.DAILY_EP, "20150107", DailyLimit("한도"))))
        self.assertEqual(self.files("kosdaq_daily"), ["date=20150102", "date=20150105", "date=20150106"])
        self.assertTrue(self.store.manifest_path.exists())
        api = FakeApi()
        self.run_c(api)  # 다음 날 다시 실행 → 남은 날만
        self.assertEqual(api.log, [(collect.DAILY_EP, "20150107"), (collect.DAILY_EP, "20150108"),
                                   (collect.DAILY_EP, "20150109")])

    def test_api_error_stops(self):
        api = FakeApi(raise_on=(collect.INDEX_EP, "20150101", ApiError("401")))
        with self.assertRaises(ApiError):
            self.run_c(api)
        self.assertEqual(api.calls, 1)

    def test_limit_and_snapshot(self):
        self.run_c(FakeApi(), limit=3)
        self.assertEqual(self.files("kosdaq_index"), ["date=20150101", "date=20150102", "date=20150105"])
        with self.assertRaises(SnapshotMismatch):
            self.run_c(FakeApi(), cfg={**CFG, "as_of": "2015-01-08"})


class FakeResp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, resps):
        self.resps, self.requests = list(resps), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.requests.append((url, params, headers))
        return self.resps.pop(0)


class ClientTest(unittest.TestCase):
    def client(self, resps, **kw):
        s = FakeSession(resps)
        return OpenApiClient("KEY", sleep_sec=0, session_factory=lambda: s, **kw), s

    def test_ok_and_request_shape(self):
        c, s = self.client([FakeResp(200, {"OutBlock_1": [{"a": "1"}]})])
        self.assertEqual(c.get("sto/ksq_bydd_trd", "20150102"), [{"a": "1"}])
        url, params, headers = s.requests[0]
        self.assertTrue(url.endswith("/svc/apis/sto/ksq_bydd_trd"))
        self.assertEqual(params, {"basDd": "20150102"})
        self.assertEqual(headers, {"AUTH_KEY": "KEY"})

    def test_401_not_retried(self):
        c, s = self.client([FakeResp(401, {"respMsg": "Unauthorized API Call", "respCode": "401"})] * 3)
        with self.assertRaisesRegex(ApiError, "Unauthorized API Call"):
            c.get("sto/ksq_bydd_trd", "20150102")
        self.assertEqual(c.calls, 1)

    def test_5xx_retried(self):
        c, _ = self.client([FakeResp(503, {}), FakeResp(200, {"OutBlock_1": []})])
        self.assertEqual(c.get("idx/kosdaq_dd_trd", "20150101"), [])  # 503 뒤 1초 쉬고 재시도
        self.assertEqual(c.calls, 2)

    def test_max_calls(self):
        c, _ = self.client([FakeResp(200, {"OutBlock_1": []})] * 3, max_calls=2)
        c.get("x", "1")
        c.get("x", "2")
        with self.assertRaises(DailyLimit):
            c.get("x", "3")


class OpenApiAdapterTest(unittest.TestCase):
    """make_legacy(종목 000001, 000002, 2015-07-13~16)에 맞춘 Open API 폴더."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.legacy, self.api = self.tmp / "legacy", self.tmp / "openapi"
        self.legacy.mkdir()
        make_legacy(self.legacy)
        (self.api / "kosdaq_index").mkdir(parents=True)
        (self.api / "kosdaq_daily").mkdir(parents=True)
        days = ["20150713", "20150714", "20150715", "20150716"]
        for i, d in enumerate(days):
            pd.DataFrame([{"IDX_NM": "코스닥 150", "OPNPRC_IDX": "1,185.83", "HGPRC_IDX": "1214.57",
                           "LWPRC_IDX": "1182.94", "CLSPRC_IDX": str(1214 + i), "ACC_TRDVOL": "1",
                           "ACC_TRDVAL": "2", "MKTCAP": "3"},
                          {"IDX_NM": "코스닥 150 소재", "OPNPRC_IDX": "", "HGPRC_IDX": "", "LWPRC_IDX": "",
                           "CLSPRC_IDX": "9", "ACC_TRDVOL": "", "ACC_TRDVAL": "", "MKTCAP": ""}]).to_parquet(
                self.api / f"kosdaq_index/date={d}.parquet")
        pd.DataFrame().to_parquet(self.api / "kosdaq_index/date=20150717.parquet")  # 휴장일(가정)
        row = lambda t, o, h, l, c, v, sh: {"ISU_CD": t, "TDD_OPNPRC": o, "TDD_HGPRC": h, "TDD_LWPRC": l,
                                            "TDD_CLSPRC": c, "ACC_TRDVOL": v, "ACC_TRDVAL": "5",
                                            "MKTCAP": "1,000", "LIST_SHRS": sh}
        daily = {
            "20150713": [row("000001", "10", "11", "9", "10", "5", "100"), row("000009", "1", "1", "1", "1", "1", "1")],
            "20150714": [row("000001", "0", "0", "0", "10", "0", "100")],                 # 거래정지
            "20150715": [row("000001", "11", "12", "10", "11", "6", "100"),
                         row("000002", "20", "21", "19", "20", "1", "50")],
            "20150716": [row("000001", "60", "61", "59", "60", "7", "100"),               # 시가 갭 +445% → 기준 변경
                         row("000002", "4", "4.5", "3.9", "4", "9", "250")],               # 주식수 5배 → 기준 변경
        }
        for d, rows in daily.items():
            pd.DataFrame(rows).to_parquet(self.api / f"kosdaq_daily/date={d}.parquet")

    def load(self, as_of="2015-07-16"):
        return openapi.load(self.api, self.legacy, as_of)

    def test_calendar_index_prices(self):
        data = self.load()
        self.assertEqual(list(data["calendar"].strftime("%Y%m%d")), ["20150713", "20150714", "20150715", "20150716"])
        self.assertEqual(data["index"]["open"].iloc[0], 1185.83)                  # 쉼표 처리
        self.assertEqual(data["index"]["close"].tolist(), [1214, 1215, 1216, 1217])  # 섹터 지수 제외
        p = data["prices"].set_index(["ticker", "date"])
        self.assertTrue(np.isnan(p.loc[("000001", pd.Timestamp("2015-07-14")), "close"]))  # 거래정지
        self.assertEqual(p.loc[("000001", pd.Timestamp("2015-07-13")), "mcap"], 1000)
        self.assertIn("000009", p.index.get_level_values(0))                       # 시장 전체 (universe 필터는 prepare에서)

    def test_corp_action_flags(self):
        p = self.load()["prices"].set_index(["ticker", "date"])["corp_action"]
        self.assertTrue(p[("000001", pd.Timestamp("2015-07-16"))])
        self.assertTrue(p[("000002", pd.Timestamp("2015-07-16"))])
        self.assertFalse(p[("000001", pd.Timestamp("2015-07-15"))])  # 직전 행이 거래정지(결측) → 판정 안 함
        self.assertEqual(int(p.sum()), 2)

    def test_members_events_from_legacy(self):
        data = self.load()
        self.assertEqual(len(data["events"]), 2)
        self.assertEqual(data["members_until"], pd.Timestamp("2015-07-16"))

    def test_early_end(self):
        (self.api / "kosdaq_daily/date=20150716.parquet").unlink()
        pd.DataFrame([{"ISU_CD": "000002", "TDD_OPNPRC": "20", "TDD_HGPRC": "21", "TDD_LWPRC": "19",
                       "TDD_CLSPRC": "20", "ACC_TRDVOL": "1", "ACC_TRDVAL": "1", "MKTCAP": "1",
                       "LIST_SHRS": "50"}]).to_parquet(self.api / "kosdaq_daily/date=20150716.parquet")
        orig = openapi.MOVED_GAP_DAYS
        openapi.MOVED_GAP_DAYS = 0
        try:
            ee = self.load()["early_end"].set_index("ticker")
        finally:
            openapi.MOVED_GAP_DAYS = orig
        # 000001: 코스닥 데이터는 07-15에 끝, 기존 데이터는 07-16까지 → moved
        self.assertEqual(ee.loc["000001", "kind"], "moved")
        self.assertNotIn("000002", ee.index)

    def test_missing_daily_file_raises(self):
        (self.api / "kosdaq_daily/date=20150715.parquet").unlink()
        with self.assertRaisesRegex(ValueError, "종목 데이터가 없는 거래일"):
            self.load()


class CorpActionTargetTest(unittest.TestCase):
    def test_corp_action_day_windows_are_nan(self):
        cal = pd.bdate_range("2022-01-03", periods=60)
        rng = np.random.default_rng(1)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
        p = pd.DataFrame({"date": cal, "ticker": "A", "open": c * 0.99, "high": c * 1.02, "low": c * 0.97,
                          "close": c, "corp_action": False})
        p.loc[30, "corp_action"] = True
        a = forward_yz(p, cal, 5).set_index("origin_date")["yz_var"]
        self.assertTrue(a.loc[cal[25]:cal[29]].isna().all())   # 구간에 30일의 야간 수익률이 들어감
        self.assertTrue(np.isfinite(a.loc[cal[24]]))
        self.assertTrue(np.isfinite(a.loc[cal[30]]))           # 구간 31~35 (o_1 = 31일 시가 / 30일 종가)


if __name__ == "__main__":
    unittest.main()
