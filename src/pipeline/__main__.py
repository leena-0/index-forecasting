"""파이프라인 준비 단계: 데이터 → 타깃 → 분할·경계 공백 → 이벤트 구간.

    python -m src.pipeline prepare [--config configs/pipeline.yaml]

결과(out_dir, git 제외):
  targets_h{h}.parquet   ticker, origin_date, horizon, yz_var, split, embargo
  event_phases.parquet   이벤트 × 구간 (complete, reason, split, crosses_split, baseline_overlaps_prev)
  meta.json              data_source, as_of, config, git commit, 패키지 버전, 건수 요약
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from . import legacy
from .target import forward_yz
from .windows import assign_splits, event_phases, summarize

REPO = Path(__file__).resolve().parents[2]


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare(cfg):
    if cfg["data_source"] != "dev_legacy":
        raise ValueError(f"지원하지 않는 data_source: {cfg['data_source']}")
    legacy_dir = cfg.get("legacy_dir") or os.environ.get("LEGACY_DATA_DIR")
    if not legacy_dir:
        sys.exit("configs/pipeline.yaml의 legacy_dir 또는 환경변수 LEGACY_DATA_DIR를 설정하세요.")
    data = legacy.load(legacy_dir, cfg["as_of"], cfg["placeholder_announce"])
    cal, prices, events = data["calendar"], data["prices"], data["events"]

    bad = legacy.check_events(events, data["members"], cal)
    print(f"거래일 {len(cal)} ({cal[0].date()} ~ {cal[-1].date()}), 종목 {prices['ticker'].nunique()}, "
          f"이벤트 {len(events)} (구성종목 변화와 안 맞음 {len(bad)})")

    out_dir = REPO / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"events": len(events), "events_mismatch": len(bad)}
    for h in cfg["horizons"]:
        t = assign_splits(forward_yz(prices, cal, h, cfg["yang_zhang"]["demean"]), cal, cfg["split"])
        usable = t["yz_var"].notna() & ~t["embargo"]
        c = {"rows": len(t), "yz_nan": int(t["yz_var"].isna().sum()), "embargo": int(t["embargo"].sum()),
             "nonpositive": int((t["yz_var"] <= 0).sum()),
             "usable_by_split": t[usable].groupby("split").size().to_dict()}
        counts[f"h{h}"] = c
        print(f"h={h}: 행 {c['rows']}, 타깃 결측 {c['yz_nan']}, 경계 공백 {c['embargo']}, "
              f"0 이하 {c['nonpositive']}, 사용 가능 {c['usable_by_split']}")
        t.to_parquet(out_dir / f"targets_h{h}.parquet", index=False)

    ph = event_phases(events, cal, prices, cfg["event_phases"], cfg["split"])
    ph.to_parquet(out_dir / "event_phases.parquet", index=False)
    s = summarize(ph)
    print("\n이벤트 구간 (split, phase, 제외 이유, 건수; 이유가 빈칸이면 사용 가능):")
    print(s.to_string(index=False))
    ev_split = ph.drop_duplicates("event_id")
    counts["events_by_split"] = ev_split.groupby("split").size().to_dict()
    counts["events_crossing_split"] = ev_split[ev_split["crosses_split"]].groupby("split").size().to_dict()
    counts["baseline_overlaps_prev"] = int(ev_split["baseline_overlaps_prev"].sum())
    print(f"\n분할별 이벤트 {counts['events_by_split']}, 그중 윈도우가 경계를 넘어 제외 "
          f"{counts['events_crossing_split']}, 기준 구간이 직전 사후 구간과 겹침 {counts['baseline_overlaps_prev']}")

    meta = {"data_source": cfg["data_source"], "as_of": cfg["as_of"], "created_at": dt.datetime.now()
            .astimezone().isoformat(timespec="seconds"), "git_commit": git_commit(), "config": cfg,
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
            "counts": counts, "note": "개발용 데이터 — 결과 수치로 보고하지 않는다"}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    print(f"\n저장: {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.pipeline")
    ap.add_argument("command", choices=["prepare"])
    ap.add_argument("--config", default=str(REPO / "configs" / "pipeline.yaml"))
    args = ap.parse_args(argv)
    load_dotenv(REPO / ".env", override=False)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prepare(cfg)


if __name__ == "__main__":
    main()
