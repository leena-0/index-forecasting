"""슬라이딩 윈도우 5개의 테스트 평가 (CLAUDE.md 4.1, 7장, 이슈 #50).

    python -m src.evaluate.walk_forward --preds data/preds/wf --data-dir data/dev --final

--preds 폴더의 w{테스트 연도}/{모델}_h{h}.parquet (src.tune --window)을 모두 읽는다. 다섯 윈도우에 같은 모델·h가 있어야 한다.
테스트 결과는 한 번만 본다: --final을 줘야 실행된다.

결과(out, git 제외):
  summary.csv, summary_by_year.csv  이어 붙인 테스트 예측의 평균 손실 (전체, 테스트 연도별)
  dm_h{h}.csv, mcs_h{h}.csv         이어 붙인 테스트 예측의 DM·MCS (QLIKE)
  sensitivity.csv                   편입 민감도 (variant, horizon, phase, model). variant: main = 주 결과,
                                    phase_min_1·base_33·base_any·with_overlap = 조건을 하나씩 바꾼 결과,
                                    stitched = 이어 붙인 예측
  event_sens.csv                    주 결과의 이벤트별 민감도
  leave_one_out.csv                 주 결과에서 이벤트를 하나씩 뺐을 때의 변화
  exclusions.csv                    variant·h·구간별 이벤트 수와 제외 이유 (이유가 빈칸이면 사용)
  no_baseline_post.csv              기준 구간 조건으로 빠진 이벤트의 사후 구간 ln(L^E / L^C)
  with_events/                      이벤트 단위 변수를 넣은 예측({모델}+EV)이 있을 때: 넣은 것과 뺀 것을 같은 공통 표본으로
                                    평가한 같은 파일들 + event_feature_effect.csv. 주 결과는 변수 없는 쪽이다 (위 파일들)
  meta.json                         입력·설정·건수, prepare meta의 reportable·hold_reasons
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.registry import EVENT_SUFFIX
from src.pipeline.__main__ import git_commit
from src.pipeline.windows import TEST_YEARS

from .compare import daily_losses, dm_table, mcs
from .event_sensitivity import (VARIANTS, aggregate, control_windows, event_losses, event_sens, leave_one_out,
                                no_baseline_post, select_events, target_phases, usage)
from .losses import loss_panel, summary

REPO = Path(__file__).resolve().parents[2]


def load_window_preds(root):
    """{테스트 연도: 예측 테이블}. 윈도우마다 (모델, h) 조합이 같아야 한다."""
    out, combos = {}, {}
    for y in TEST_YEARS:
        files = sorted((Path(root) / f"w{y}").glob("*.parquet"))
        if not files:
            raise ValueError(f"w{y}에 예측 테이블이 없습니다: {root}")
        out[y] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        combos[y] = set(map(tuple, out[y][["model", "horizon"]].drop_duplicates().to_numpy()))
    if len({frozenset(c) for c in combos.values()}) != 1:
        raise ValueError(f"윈도우마다 모델·h가 다릅니다: { {y: sorted(c) for y, c in combos.items()} }")
    return out


def years(df, *ys):
    return df[df["origin_date"].dt.year.isin(ys)]


def test_targets(targets):
    """고정 분할의 split·embargo를 지운 타깃. 달력 밖으로 나가는 타깃은 yz_var가 결측이라 빠진다."""
    return targets.assign(split="test", embargo=False)


def split_event_preds(preds):
    """(변수 없는 예측, 모든 예측 또는 None). 변수를 넣은 모델마다 변수 없는 같은 모델이 있어야 한다."""
    base = {y: p[~p["model"].str.endswith(EVENT_SUFFIX)] for y, p in preds.items()}
    names = {m for p in preds.values() for m in p["model"].unique()}
    missing = {m for m in names if m.endswith(EVENT_SUFFIX) and m.removesuffix(EVENT_SUFFIX) not in names}
    if missing:
        raise ValueError(f"변수 없는 예측이 없습니다: {sorted(missing)}")
    return base, (preds if any(m.endswith(EVENT_SUFFIX) for m in names) else None)


def event_feature_effect(out_dir, horizons):
    """같은 모델의 변수 유무 비교: 평균 QLIKE 차이(없음 − 있음, > 0이면 변수를 넣은 쪽의 손실이 작다)와 DM 검정."""
    rows = []
    for h in horizons:
        dm = pd.read_csv(out_dir / f"dm_h{h}.csv")
        rows.append(dm[dm["model_2"] == dm["model_1"] + EVENT_SUFFIX].assign(horizon=h))
    t = pd.concat(rows, ignore_index=True).rename(columns={"model_1": "model", "mean_diff": "qlike_without_minus_with"})
    return t[["horizon", "model", "qlike_without_minus_with", "stat", "p_value", "n"]]


def evaluate(preds, data_dir, out_dir, alpha=0.10, B=1000, seed=0):
    prep_meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    phases = pd.read_parquet(data_dir / "event_phases.parquet")
    members = pd.read_parquet(data_dir / "members.parquet")
    calendar = pd.DatetimeIndex(np.sort(pd.read_parquet(data_dir / "features.parquet", columns=["date"])["date"].unique()))
    events = select_events(phases, calendar)
    used = events[events["excluded"] == ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"events": events["excluded"].replace("", "사용").value_counts().to_dict(),
              "events_by_window": used["window"].value_counts().sort_index().to_dict()}
    panels, sens, per_event, loo, excl, nbp = [], [], [], [], [], []
    for h in sorted(preds[TEST_YEARS[0]]["horizon"].unique()):
        targets = test_targets(pd.read_parquet(data_dir / f"targets_h{h}.parquet"))
        by_window = {y: p[p["horizon"] == h] for y, p in preds.items()}
        stitched, c = loss_panel(pd.concat([years(p, y) for y, p in by_window.items()]), targets)
        counts[f"h{h}"] = c
        panels.append(stitched)
        daily = daily_losses(stitched)
        dm_table(daily, h).to_csv(out_dir / f"dm_h{h}.csv", index=False)
        mcs(daily, alpha=alpha, B=B, block=h, seed=seed).to_csv(out_dir / f"mcs_h{h}.csv", index=False)

        shifted, control = target_phases(phases, calendar, h), control_windows(phases, calendar, h)
        losses = pd.concat([event_losses(loss_panel(years(p, y, y + 1), targets)[0], members, control, shifted,
                                         used[used["window"] == y]) for y, p in by_window.items()], ignore_index=True)
        st_losses = event_losses(stitched, members, control, shifted, used)
        runs = [(v, losses, cfg) for v, cfg in VARIANTS.items()] + [("stitched", st_losses, VARIANTS["main"])]
        for name, ls, cfg in runs:
            use = usage(shifted, events, ls, *cfg)
            es = event_sens(ls, use)
            sens.append(aggregate(es, events, B, seed).assign(variant=name, horizon=h))
            excl.append(use[use["phase"] != "baseline"].groupby(["phase", "reason"]).size().rename("n_events")
                        .reset_index().assign(variant=name, horizon=h))
            if name == "main":
                per_event.append(es.merge(events[["event_id", "ticker", "event_type", "effective", "window"]],
                                          on="event_id").assign(horizon=h))
                loo.append(leave_one_out(es).assign(horizon=h))
                nbp.append(no_baseline_post(ls, use, cfg[1]).assign(horizon=h))
    panel = pd.concat(panels, ignore_index=True)
    summary(panel).to_csv(out_dir / "summary.csv", index=False)
    by_year = panel.assign(split=panel["origin_date"].dt.year)
    summary(by_year).rename(columns={"split": "test_year"}).to_csv(out_dir / "summary_by_year.csv", index=False)
    keys = ["variant", "horizon"]
    for frames, name in [(sens, "sensitivity"), (per_event, "event_sens"), (loo, "leave_one_out"),
                         (excl, "exclusions"), (nbp, "no_baseline_post")]:
        t = pd.concat(frames, ignore_index=True)
        first = [k for k in keys if k in t]
        t[first + [x for x in t.columns if x not in first]].to_csv(out_dir / f"{name}.csv", index=False)

    models = {m: sorted(map(int, g.unique())) for m, g in pd.concat(preds.values()).groupby("model")["seed"]}
    meta = {"split": "walk_forward_test", "test_years": list(TEST_YEARS), "data_dir": str(data_dir),
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "git_commit": git_commit(),
            "models": sorted(models), "seeds": models,
            "pred_versions": sorted(pd.concat(preds.values())["version"].astype(str).unique()),
            "settings": {"loss_for_tests": "qlike", "dm_hac_lag": "h-1", "mcs_alpha": alpha, "B": B, "mcs_block": "h",
                         "seed": seed, "common_sample": True, "variants": VARIANTS,
                         "phase_assignment": "target window inside phase", "ci": "effective-date bootstrap, 95%"},
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
            "counts": counts, "prepare_meta": {k: prep_meta.get(k) for k in
                                               ["data_source", "as_of", "git_commit", "reportable", "hold_reasons"]},
            "reportable": bool(prep_meta.get("reportable", False))}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"편입 이벤트 {counts['events']}, 윈도우별 {counts['events_by_window']}")
    if not meta["reportable"]:
        print(f"[보류] prepare 결과가 reportable=false: {prep_meta.get('hold_reasons')}")
    print(f"저장: {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.evaluate.walk_forward")
    ap.add_argument("--preds", default="data/preds/wf", help="w{테스트 연도}/ 폴더들이 있는 예측 폴더")
    ap.add_argument("--data-dir", default="data/dev", help="src.pipeline prepare 결과 폴더")
    ap.add_argument("--out", default="data/eval/walk_forward")
    ap.add_argument("--final", action="store_true", help="테스트 구간 평가를 허용 (한 번만)")
    ap.add_argument("--alpha", type=float, default=0.10, help="MCS 유의수준")
    ap.add_argument("--B", type=int, default=1000, help="부트스트랩 횟수 (MCS, 민감도 신뢰구간)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.final:
        sys.exit("테스트 구간은 한 번만 봅니다 (CLAUDE.md 4.1). 규칙과 튜닝이 끝났으면 --final을 주세요.")
    base, both = split_event_preds(load_window_preds(REPO / args.preds))
    data_dir, out = REPO / args.data_dir, REPO / args.out
    evaluate(base, data_dir, out, args.alpha, args.B, args.seed)
    if both is not None:
        evaluate(both, data_dir, out / "with_events", args.alpha, args.B, args.seed)
        effect = event_feature_effect(out / "with_events", sorted(both[TEST_YEARS[0]]["horizon"].unique()))
        effect.to_csv(out / "with_events" / "event_feature_effect.csv", index=False)


if __name__ == "__main__":
    main()
