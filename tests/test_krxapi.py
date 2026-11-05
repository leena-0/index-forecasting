"""src.krxapi(수집)와 src.pipeline.openapi(어댑터) 단위 테스트 — 가짜 클라이언트·가짜 Session, 네트워크 없음."""
import datetime as dt
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.krxapi import collect
from src.krxapi.client import ApiError, DailyLimit, OpenApiClient, RequestBudget
from src.krxapi.store import SnapshotMismatch, Store
from src.pipeline import openapi
from src.pipeline.target import forward_yz
from tests.test_pipeline import make_legacy

CFG = {"as_of": "2015-01-09", "start": "2015-01-01"}
HOLIDAYS = {"20150101"}
WEEKDAYS = ["20150101", "20150102", "20150105", "20150106", "20150107", "20150108", "20150109"]


def idx_row(d, name):
    return {"BAS_DD": d, "IDX_NM": name, "OPNPRC_IDX": "1", "HGPRC_IDX": "1", "LWPRC_IDX": "1",
            "CLSPRC_IDX": "1,000.5", "ACC_TRDVOL": "1", "ACC_TRDVAL": "1"}


def daily_row(d, code="000001", o="100", h="110", l="90", c="105", v="10", cmp="5", fluc="5.00"):
    return {"BAS_DD": d, "ISU_CD": code, "ISU_NM": "가", "TDD_OPNPRC": o, "TDD_HGPRC": h, "TDD_LWPRC": l,
            "TDD_CLSPRC": c, "CMPPREVDD_PRC": cmp, "FLUC_RT": fluc,
            "ACC_TRDVOL": v, "ACC_TRDVAL": "1", "MKTCAP": "1", "LIST_SHRS": "1"}


class FakeBudget:
    def used(self):
        return 0


class FakeApi:
    """endpoint별로 정해 둔 응답. 휴장일은 지수·종목 모두 빈 응답. override[(endpoint, 날짜)] = 응답."""

    def __init__(self, override=None, raise_on=None):
        self.calls = 0
        self.log = []
        self.override = override or {}
        self.raise_on = raise_on
        self.budget = FakeBudget()

    def get(self, endpoint, bas_dd):
        self.calls += 1
        self.log.append((endpoint, bas_dd))
        if self.raise_on and self.raise_on[0] == endpoint and bas_dd >= self.raise_on[1]:
            raise self.raise_on[2]
        if (endpoint, bas_dd) in self.override:
            return self.override[(endpoint, bas_dd)]
        if bas_dd in HOLIDAYS and endpoint != collect.BASIC_EP:
            return []
        if endpoint == collect.INDEX_EP:
            return [idx_row(bas_dd, "코스닥"), idx_row(bas_dd, "코스닥 150")]
        if endpoint == collect.BASIC_EP:
            return [{"ISU_CD": "KR7000001", "ISU_SRT_CD": "000001", "ISU_NM": "가", "LIST_DD": "20100104",
                     "LIST_SHRS": "1"}]
        return [daily_row(bas_dd)]


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.store = Store(self.tmp, "tester")

    def run_c(self, api, limit=None, cfg=CFG, only_date=None):
        return collect.Collector(api, self.store, cfg, log=lambda *_: None).run(limit=limit, only_date=only_date)

    def files(self, kind):
        return sorted(p.stem[5:] for p in (self.tmp / "raw/openapi" / kind).glob("*.parquet"))

    def test_holidays_and_trading_days(self):
        api = FakeApi()
        self.assertEqual(self.run_c(api), 0)
        self.assertEqual(self.files("kosdaq_index"), WEEKDAYS)
        self.assertEqual(self.files("kosdaq_daily"), WEEKDAYS)                 # 휴장일은 0행 표시 파일
        self.assertEqual(len(self.store.read("raw/openapi/kosdaq_daily/date=20150101.parquet")), 0)
        self.assertIn((collect.DAILY_EP, "20150101"), api.log)                  # 휴장일 이중 확인
        self.assertTrue(all(d <= "20150109" for _, d in api.log))               # as_of 이후 요청 안 함
        self.assertEqual(self.files("kosdaq_basic"), ["20150109"])              # 달마다 마지막 거래일
        self.assertEqual(collect.validate_files(self.store, CFG, complete=True), [])

    def test_empty_index_but_stock_data_is_failure(self):
        api = FakeApi(override={(collect.INDEX_EP, "20150106"): []})
        self.assertEqual(self.run_c(api), 1)
        self.assertNotIn("20150106", self.files("kosdaq_index"))
        self.assertNotIn("20150106", self.files("kosdaq_daily"))

    def test_response_checks(self):
        bad = {(collect.DAILY_EP, "20150105"): [daily_row("20150102")],                       # 다른 날짜
               (collect.DAILY_EP, "20150106"): [daily_row("20150106"), daily_row("20150106")],  # 종목코드 중복
               (collect.DAILY_EP, "20150107"): [],                                              # 거래일 0행
               (collect.INDEX_EP, "20150108"): [idx_row("20150108", "코스닥")]}                 # 코스닥 150 없음
        self.assertEqual(self.run_c(FakeApi(override=bad)), 1)
        self.assertEqual(self.files("kosdaq_daily"), ["20150101", "20150102", "20150109"])
        api = FakeApi()
        self.assertEqual(self.run_c(api), 0)                                    # 다시 실행 → 실패한 것만
        self.assertEqual(sorted(api.log), sorted([(collect.INDEX_EP, "20150108"), (collect.DAILY_EP, "20150105"),
                                                  (collect.DAILY_EP, "20150106"), (collect.DAILY_EP, "20150107"),
                                                  (collect.DAILY_EP, "20150108")]))

    def test_rerun_skips(self):
        self.run_c(FakeApi())
        api = FakeApi()
        self.run_c(api)
        self.assertEqual(api.calls, 0)

    def test_daily_limit_returns_2_and_resumes(self):
        code = self.run_c(FakeApi(raise_on=(collect.DAILY_EP, "20150107", DailyLimit("한도"))))
        self.assertEqual(code, 2)
        self.assertEqual(self.files("kosdaq_daily"), ["20150101", "20150102", "20150105", "20150106"])
        self.assertTrue(self.store.manifest_path.exists())
        api = FakeApi()
        self.assertEqual(self.run_c(api), 0)  # 다음 날 다시 실행 → 남은 것만
        self.assertEqual(api.log, [(collect.DAILY_EP, "20150107"), (collect.DAILY_EP, "20150108"),
                                   (collect.DAILY_EP, "20150109"), (collect.BASIC_EP, "20150109")])

    def test_api_error_stops(self):
        api = FakeApi(raise_on=(collect.INDEX_EP, "20150101", ApiError("401")))
        with self.assertRaises(ApiError):
            self.run_c(api)
        self.assertEqual(api.calls, 1)

    def test_month_ends(self):
        self.assertEqual(collect.month_ends(["20150129", "20150130", "20150202", "20150227", "20150302"]),
                         ["20150130", "20150227", "20150302"])

    def test_limit_date_and_snapshot(self):
        self.run_c(FakeApi(), limit=3)
        self.assertEqual(self.files("kosdaq_index"), ["20150101", "20150102", "20150105"])
        self.run_c(FakeApi(), only_date="2015-01-08")
        self.assertIn("20150108", self.files("kosdaq_daily"))
        with self.assertRaises(ValueError):
            self.run_c(FakeApi(), only_date="2015-01-10")                       # 토요일
        with self.assertRaises(SnapshotMismatch):
            self.run_c(FakeApi(), cfg={**CFG, "as_of": "2015-01-08"})

    def test_validate_finds_problems(self):
        self.run_c(FakeApi())
        (self.tmp / "raw/openapi/kosdaq_daily/date=20150105.parquet").unlink()
        pd.DataFrame([daily_row("20150106", l="120")]).to_parquet(              # 저가 > 고가
            self.tmp / "raw/openapi/kosdaq_daily/date=20150106.parquet")
        pd.DataFrame([daily_row("20150107", o="0", h="0", l="0", c="99", v="0")]).to_parquet(  # 거래정지 행은 OK
            self.tmp / "raw/openapi/kosdaq_daily/date=20150107.parquet")
        (self.tmp / "raw/openapi/kosdaq_index/date=20150108.parquet").unlink()
        issues = collect.validate_files(self.store, CFG, complete=True)
        text = "\n".join(issues)
        self.assertIn("20150105: 거래일 종목 파일 없음", text)
        self.assertIn("20150106: OHLC 관계 오류", text)
        self.assertNotIn("20150107", text)
        self.assertIn("평일 지수 파일 1개 누락", text)
        self.assertEqual(len(issues), 3)


class FakeResp:
    def __init__(self, status, data, headers=None):
        self.status_code, self._data, self.headers = status, data, headers or {}

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
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
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def client(self, resps, limit=100, today=lambda: dt.date(2026, 10, 19)):
        s = FakeSession(resps)
        budget = RequestBudget(self.tmp, limit=limit, today=today)
        return OpenApiClient("KEY", budget, sleep_sec=0, session_factory=lambda: s), s

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

    def test_429_stops(self):
        c, _ = self.client([FakeResp(429, {}, {"Retry-After": "3600"})] * 3)
        with self.assertRaisesRegex(DailyLimit, "429"):
            c.get("x", "1")
        self.assertEqual(c.calls, 1)

    def test_5xx_and_bad_json_retried(self):
        c, _ = self.client([FakeResp(503, {}), FakeResp(200, ValueError("html")), FakeResp(200, {"OutBlock_1": []})])
        self.assertEqual(c.get("idx/kosdaq_dd_trd", "20150101"), [])  # 재시도 사이 1초, 2초 쉼
        self.assertEqual(c.calls, 3)

    def test_budget_is_shared_across_runs_and_resets_by_day(self):
        c1, _ = self.client([FakeResp(200, {"OutBlock_1": []})] * 2, limit=3)
        c1.get("x", "1")
        c1.get("x", "2")
        c2, _ = self.client([FakeResp(200, {"OutBlock_1": []})] * 2, limit=3)   # 같은 날 다시 실행
        c2.get("x", "3")
        with self.assertRaises(DailyLimit):
            c2.get("x", "4")
        self.assertEqual(c2.budget.used(), 3)
        c3, _ = self.client([FakeResp(200, {"OutBlock_1": []})], limit=3, today=lambda: dt.date(2026, 10, 20))
        c3.get("x", "5")                                                          # 다음 날은 새로 셈
        self.assertEqual(c3.budget.used(), 1)


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
        row = lambda t, o, h, l, c, v, sh, cmp, fluc: {
            "ISU_CD": t, "TDD_OPNPRC": o, "TDD_HGPRC": h, "TDD_LWPRC": l, "TDD_CLSPRC": c,
            "CMPPREVDD_PRC": cmp, "FLUC_RT": fluc, "ACC_TRDVOL": v, "ACC_TRDVAL": "5",
            "MKTCAP": "1,000", "LIST_SHRS": sh}
        daily = {
            "20150713": [row("000001", "10", "11", "9", "10", "5", "100", "0", "0"),
                         row("000009", "1", "1", "1", "1", "1", "1", "0", "0")],
            "20150714": [row("000001", "0", "0", "0", "10", "0", "100", "0", "0")],  # 거래정지
            "20150715": [row("000001", "11", "12", "10", "11", "6", "100", "1", "10"),
                         row("000002", "20", "21", "19", "20", "1", "50", "0", "0")],
            "20150716": [row("000001", "60", "61", "59", "60", "7", "100", "49", "445.45"),
                         row("000002", "4", "4.5", "3.9", "4", "9", "250", "0", "0")],
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
        self.assertEqual(p.loc[("000001", pd.Timestamp("2015-07-15")), "cmp_prevdd_prc"], 1)
        self.assertEqual(p.loc[("000001", pd.Timestamp("2015-07-15")), "base_price"], 10)
        self.assertIn("000009", p.index.get_level_values(0))                       # 시장 전체 (universe 필터는 prepare에서)

    def test_corp_action_flags(self):
        p = self.load()["prices"].set_index(["ticker", "date"])["corp_action"]
        self.assertTrue(p[("000001", pd.Timestamp("2015-07-16"))])
        self.assertTrue(p[("000002", pd.Timestamp("2015-07-16"))])
        self.assertFalse(p[("000001", pd.Timestamp("2015-07-15"))])  # 직전 행이 거래정지(결측) → 판정 안 함
        self.assertEqual(int(p.sum()), 2)

    def test_base_price_validity_and_corp_action_recovery(self):
        p = self.load()["prices"].set_index(["ticker", "date"])
        first = ("000002", pd.Timestamp("2015-07-15"))
        adjusted = ("000001", pd.Timestamp("2015-07-16"))
        halt = ("000001", pd.Timestamp("2015-07-14"))
        self.assertFalse(p.loc[first, "base_price_valid"])       # 종목 첫 관측일
        self.assertFalse(p.loc[halt, "base_price_valid"])        # 거래정지일
        after_halt = ("000001", pd.Timestamp("2015-07-15"))
        self.assertFalse(p.loc[after_halt, "base_price_valid"])  # 거래정지 직후 (CLAUDE.md 5.3)
        self.assertTrue(p.loc[after_halt, "base_price_after_halt"])
        self.assertEqual(p.loc[after_halt, "base_price"], 10)    # 기준가 값은 남긴다
        self.assertTrue(p.loc[adjusted, "base_price_valid"])     # 기존 규칙은 잡지만 기준가는 복원 가능
        self.assertTrue(p.loc[adjusted, "corp_action"])

    def test_members_events_from_legacy(self):
        data = self.load()
        self.assertEqual(len(data["events"]), 2)
        self.assertEqual(data["members_until"], pd.Timestamp("2015-07-16"))

    def test_early_end(self):
        (self.api / "kosdaq_daily/date=20150716.parquet").unlink()
        pd.DataFrame([{"ISU_CD": "000002", "TDD_OPNPRC": "20", "TDD_HGPRC": "21", "TDD_LWPRC": "19",
                       "TDD_CLSPRC": "20", "CMPPREVDD_PRC": "0", "FLUC_RT": "0", "ACC_TRDVOL": "1",
                       "ACC_TRDVAL": "1", "MKTCAP": "1",
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

    def test_listing_dates(self):
        (self.api / "kosdaq_basic").mkdir()
        pd.DataFrame([{"ISU_SRT_CD": "000001", "LIST_DD": "20100104"},
                      {"ISU_SRT_CD": "000002", "LIST_DD": "20150715"}]).to_parquet(
            self.api / "kosdaq_basic/date=20150716.parquet")
        pd.DataFrame([{"ISU_SRT_CD": "000003", "LIST_DD": "20200101"}]).to_parquet(
            self.api / "kosdaq_basic/date=20200131.parquet")                          # as_of 이후 파일은 안 읽음
        lst = self.load()["listing"].set_index("ticker")["list_date"]
        self.assertEqual(lst.to_dict(), {"000001": pd.Timestamp("2010-01-04"), "000002": pd.Timestamp("2015-07-15")})

    def test_missing_daily_file_raises(self):
        (self.api / "kosdaq_daily/date=20150715.parquet").unlink()
        with self.assertRaisesRegex(ValueError, "종목 데이터가 없는 거래일"):
            self.load()


class CorpActionRuleTest(unittest.TestCase):
    """판정: ratio = 시가 / 전일 종가, ratio > 1.3 또는 ratio < 0.7 (경계값은 포함하지 않음)."""

    def flags(self, opens, shares=None):
        n = len(opens) + 1
        p = pd.DataFrame({"ticker": "A", "date": pd.bdate_range("2022-01-03", periods=n),
                          "open": [100.0] + list(opens), "close": 100.0,
                          "shares": shares if shares is not None else [1000.0] * n})
        return openapi.flag_corp_actions(p)["corp_action"].tolist()[1:]  # 전일 종가는 모두 100

    def test_price_ratio_bounds(self):
        opens = [131, 130, 129, 71, 70, 69, 75]
        self.assertEqual(self.flags(opens), [True, False, False, False, False, True, False])
        # 0.75: 이전 기준(|ln ratio| > ln 1.3, 하한 ≈ 0.769)에서는 걸렸지만 새 기준에서는 가격제한폭 안이라 걸리지 않음

    def test_shares_ratio_still_applies(self):
        self.assertEqual(self.flags([100, 100], shares=[1000.0, 1000.0, 1600.0]), [False, True])

    def test_custom_gap_limit(self):
        p = pd.DataFrame({"ticker": "A", "date": pd.bdate_range("2022-01-03", periods=2),
                          "open": [100.0, 116.0], "close": 100.0, "shares": 1.0})
        self.assertTrue(openapi.flag_corp_actions(p, gap_limit=0.15)["corp_action"].iloc[1])  # 2015-06 이전 ±15%
        self.assertFalse(openapi.flag_corp_actions(p)["corp_action"].iloc[1])


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


class BasePriceTest(unittest.TestCase):
    def prices(self, n=60):
        cal = pd.bdate_range("2022-01-03", periods=n)
        close = 100 * np.exp(np.arange(n) * 0.001)
        base = np.r_[close[0], close[:-1]]
        p = pd.DataFrame({"date": cal, "ticker": "A", "open": base * 1.001,
                          "high": base * 1.012, "low": base * 0.992, "close": close,
                          "volume": 100.0, "shares": 1000.0, "mcap": close * 1000,
                          "cmp_prevdd_prc": close - base,
                          "fluc_rt": 100 * (close - base) / base})
        p["base_price"] = base
        p["base_price_valid"] = True
        p.loc[0, "base_price_valid"] = False
        return cal, p

    def test_normal_days_match_previous_close_method(self):
        cal, p = self.prices()
        legacy = p.drop(columns=["cmp_prevdd_prc", "fluc_rt", "base_price", "base_price_valid"])
        old = forward_yz(legacy, cal, 5).set_index("origin_date")["yz_var"]
        new = forward_yz(p, cal, 5).set_index("origin_date")["yz_var"]
        pd.testing.assert_series_equal(old.iloc[1:-5], new.iloc[1:-5])

    def test_split_like_jump_uses_base_price_instead_of_dropping_window(self):
        cal, p = self.prices()
        i = 30
        p.loc[i, ["open", "high", "low", "close"]] = [50, 53, 49, 52]
        p.loc[i, ["cmp_prevdd_prc", "base_price"]] = [2, 50]
        p.loc[i, "fluc_rt"] = 4
        p["corp_action"] = False
        p.loc[i, "corp_action"] = True
        new = forward_yz(p, cal, 5).set_index("origin_date")["yz_var"]
        self.assertTrue(np.isfinite(new.loc[cal[i - 5:i]]).all())

        old = p.drop(columns=["cmp_prevdd_prc", "fluc_rt", "base_price", "base_price_valid"])
        dropped = forward_yz(old, cal, 5).set_index("origin_date")["yz_var"]
        self.assertTrue(dropped.loc[cal[i - 5:i]].isna().all())

    def test_invalid_formula_halt_first_observation_and_listing(self):
        cal, p = self.prices(n=7)
        listing = pd.DataFrame({"ticker": ["A"], "list_date": [cal[3]]})
        p = p.drop(columns=["base_price", "base_price_valid"])
        p.loc[1, "cmp_prevdd_prc"] = p.loc[1, "close"]  # base_price = 0
        p.loc[2, "fluc_rt"] += 1
        p.loc[4, ["open", "high", "low", "volume"]] = 0
        p.loc[5, "cmp_prevdd_prc"] = np.nan
        out = openapi.add_base_prices(p, listing)
        self.assertFalse(out.loc[0, "base_price_valid"])  # 첫 관측일
        self.assertFalse(out.loc[1, "base_price_valid"])  # 0 이하 기준가
        self.assertFalse(out.loc[2, "base_price_valid"])  # 내부 검증식 불일치
        self.assertFalse(out.loc[3, "base_price_valid"])  # 신규상장일
        self.assertFalse(out.loc[4, "base_price_valid"])  # 거래정지일
        self.assertFalse(out.loc[5, "base_price_valid"])  # 거래정지 직후 (결측 기준가이기도 함)
        self.assertTrue(out.loc[5, "base_price_after_halt"])
        self.assertTrue(out.loc[6, "base_price_valid"])

    def test_day_after_missing_row_counts_as_after_halt(self):
        cal, p = self.prices(n=7)
        p = p.drop(columns=["base_price", "base_price_valid"]).drop(index=3).reset_index(drop=True)
        out = openapi.add_base_prices(p, calendar=cal).set_index("date")
        self.assertFalse(out.loc[cal[4], "base_price_valid"])  # 달력상 직전 거래일 행이 없음
        self.assertTrue(out.loc[cal[5], "base_price_valid"])
        no_cal = openapi.add_base_prices(p).set_index("date")
        self.assertTrue(no_cal.loc[cal[4], "base_price_valid"])

    def test_after_halt_day_is_not_in_target(self):
        cal, p = self.prices()
        p = p.drop(columns=["base_price", "base_price_valid"])
        p.loc[30, ["open", "high", "low", "volume"]] = np.nan
        p.loc[30, "close"] = np.nan
        out = openapi.add_base_prices(p, calendar=cal)
        a = forward_yz(out, cal, 5).set_index("origin_date")["yz_var"]
        self.assertTrue(a.loc[cal[26]:cal[30]].isna().all())  # 구간에 30일 또는 31일이 들어감
        self.assertTrue(np.isfinite(a.loc[cal[31]]))


if __name__ == "__main__":
    unittest.main()
