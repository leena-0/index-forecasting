"""공통 평가: 예측 테이블(CLAUDE.md 6.2)만 읽어 지표를 계산한다 (7장).

    python -m src.evaluate --pred 예측1.parquet [예측2.parquet ...] --data-dir data/dev_openapi --split val

data-dir는 src.pipeline prepare의 결과 폴더 (targets_h{h}.parquet, event_phases.parquet, members.parquet, meta.json).
테스트 구간은 한 번만 본다 (CLAUDE.md 4.1): --split test는 --final을 함께 줘야 실행된다.

결과(out, git 제외):
  summary.csv            split, horizon, model, n, qlike, mse
  dm_h{h}.csv            모델 쌍별 Diebold–Mariano (QLIKE, HAC 시차 h−1)
  mcs_h{h}.csv           Model Confidence Set (QLIKE, T_max)
  sensitivity.csv        편입 민감도 Sens_{m,s} (QLIKE)
  no_baseline_post.csv   기준 구간 없는 편입 이벤트의 사후 구간 ln(L^E / L^C)
  meta.json              입력·설정·건수, prepare meta의 reportable·hold_reasons를 그대로 옮김
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.__main__ import git_commit

from .compare import daily_losses, dm_table, mcs
from .losses import loss_panel, summary
from .sensitivity import event_control_losses, no_baseline_post, sensitivity

REPO = Path(__file__).resolve().parents[2]


def evaluate(pred, data_dir, split, out_dir, alpha=0.10, B=1000, seed=0):
    prep_meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    phases = pd.read_parquet(data_dir / "event_phases.parquet")
    members = pd.read_parquet(data_dir / "members.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, panels = {}, []
    for h in sorted(pred["horizon"].unique()):
        targets = pd.read_parquet(data_dir / f"targets_h{h}.parquet")
        p, c = loss_panel(pred[pred["horizon"] == h], targets[targets["split"] == split])
        counts[f"h{h}"] = c
        print(f"h={h}: 예측 {c['pred_rows']}행, 다른 분할·타깃 없음·경계 공백·0 이하 {c['no_usable_target']}, "
              f"공통 집합 밖 {c['dropped_not_common']}, 평가 {c['rows']}")
        if p.empty:
            continue
        panels.append(p)
        daily = daily_losses(p)
        dm_table(daily, h).to_csv(out_dir / f"dm_h{h}.csv", index=False)
        mcs(daily, alpha=alpha, B=B, block=h, seed=seed).to_csv(out_dir / f"mcs_h{h}.csv", index=False)
    if not panels:
        sys.exit(f"{split} 구간에 평가할 행이 없습니다.")
    panel = pd.concat(panels, ignore_index=True)
    s = summary(panel)
    s.to_csv(out_dir / "summary.csv", index=False)
    print(s.to_string(index=False))

    sens, nbp = [], []
    for h, p in panel.groupby("horizon"):
        ec = event_control_losses(p, members, phases)
        sens.append(sensitivity(ec).assign(horizon=h))
        nbp.append(no_baseline_post(ec).assign(horizon=h))
    sens, nbp = pd.concat(sens, ignore_index=True), pd.concat(nbp, ignore_index=True)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    nbp.to_csv(out_dir / "no_baseline_post.csv", index=False)
    counts["sensitivity_events"] = sens.groupby(["horizon", "phase"])["n_events"].max().rename("n") \
        .reset_index().to_dict("records")
    print(f"\n편입 민감도 이벤트 수 (구간별): {counts['sensitivity_events']}")

    meta = {"split": split, "data_dir": str(data_dir), "created_at": dt.datetime.now().astimezone()
            .isoformat(timespec="seconds"), "git_commit": git_commit(),
            "models": sorted(pred["model"].unique()), "pred_versions": sorted(pred["version"].astype(str).unique()),
            "seeds": {m: sorted(map(int, g.unique())) for m, g in pred.groupby("model")["seed"]},
            "settings": {"loss_for_tests": "qlike", "dm_hac_lag": "h-1", "mcs_alpha": alpha, "mcs_B": B,
                         "mcs_block": "h", "mcs_seed": seed, "common_sample": True},
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
            "counts": counts, "prepare_meta": {k: prep_meta.get(k) for k in
                                               ["data_source", "as_of", "git_commit", "reportable", "hold_reasons"]},
            "reportable": bool(prep_meta.get("reportable", False))}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not meta["reportable"]:
        print(f"\n[보류] prepare 결과가 reportable=false: {prep_meta.get('hold_reasons')}")
    print(f"저장: {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.evaluate")
    ap.add_argument("--pred", nargs="+", required=True, help="예측 테이블 parquet (CLAUDE.md 6.2)")
    ap.add_argument("--data-dir", default="data/dev_openapi", help="src.pipeline prepare 결과 폴더")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="테스트 구간 평가를 허용 (한 번만)")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본 data/eval/<split>)")
    ap.add_argument("--alpha", type=float, default=0.10, help="MCS 유의수준")
    ap.add_argument("--B", type=int, default=1000, help="MCS 부트스트랩 횟수")
    ap.add_argument("--seed", type=int, default=0, help="MCS 부트스트랩 seed")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.final:
        sys.exit("테스트 구간은 한 번만 봅니다 (CLAUDE.md 4.1). 모델·예측 기간 선택이 끝났으면 --final을 주세요.")
    pred = pd.concat([pd.read_parquet(p) for p in args.pred], ignore_index=True)
    out = Path(args.out) if args.out else REPO / "data" / "eval" / args.split
    evaluate(pred, REPO / args.data_dir, args.split, out, args.alpha, args.B, args.seed)


if __name__ == "__main__":
    main()
