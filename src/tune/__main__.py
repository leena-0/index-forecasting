"""검증 구간 하이퍼파라미터 튜닝 (docs/TUNING.md).

    python -m src.tune --model lgbm --h 5 [--grid configs/tuning.yaml] [--data-dir data/dev] [--jobs 4]

결과 (git 제외):
  data/tuning/{model}_h{h}/results.csv     설정 × 편향 보정별 검증 지표 (qlike_val 순위)
  data/tuning/{model}_h{h}/results.md      같은 표 (Markdown)
  data/tuning/{model}_h{h}/val_preds/{config_id}.parquet   설정별 검증 구간 예측값 (보정 전·후)
  data/preds/tuned/{모델}_h{h}.parquet     가장 좋은 설정의 전체 예측 테이블 (CLAUDE.md 6.2) + .meta.json
테스트 구간 지표는 계산하지 않는다. 고른 설정의 테스트 평가는 src.evaluate --split test --final로 한 번만 한다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models import registry
from src.models.__main__ import prediction_table
from src.pipeline.__main__ import git_commit
from src.pipeline.features import load_panel

from .runner import expand, run_configs, score

REPO = Path(__file__).resolve().parents[2]


def to_markdown(table):
    cols = ["rank", "config_id", "smearing", "smearing_c", "qlike_val", "mse_val", "qlike_event", "n_val", "n_event"]
    t = table[cols].copy()
    for c in ["qlike_val", "qlike_event"]:
        t[c] = t[c].map(lambda v: f"{v:.5f}")
    t["mse_val"] = t["mse_val"].map(lambda v: f"{v:.3e}")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def tune(model, h, data_dir, grid, out_root, preds_dir, seed=0, jobs=1):
    configs = expand(grid)
    print(f"[tune] {model} h={h}: 설정 {len(configs)}개 × 편향 보정 2가지, jobs={jobs}")
    results = run_configs(model, data_dir, h, configs, seed, jobs)
    panel = load_panel(data_dir, h)
    phases = pd.read_parquet(data_dir / "event_phases.parquet")
    table, mask = score(panel, phases, results, configs)

    out = out_root / f"{model}_h{h}"
    (out / "val_preds").mkdir(parents=True, exist_ok=True)
    keys = panel.loc[mask, ["ticker", "origin_date"]].reset_index(drop=True)
    for cid, pred, c, _ in results:
        keys.assign(pred_log_var=pred[mask], pred_log_var_smeared=pred[mask] + c).to_parquet(
            out / "val_preds" / f"{cid}.parquet", index=False)
    prep_meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    header = (f"# {registry.DISPLAY[model]} h={h} 튜닝 (검증 구간, {prep_meta.get('config', {}).get('split')})\n\n"
              f"개발용 — reportable={prep_meta.get('reportable')}, hold_reasons={prep_meta.get('hold_reasons')}\n\n")
    table.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(header + to_markdown(table), encoding="utf-8")

    best = table.iloc[0]
    i = [r[0] for r in results].index(best["config_id"])
    cid, pred, c, info = results[i]
    lv = pred.astype(float) + (c if best["smearing"] else 0.0)
    name = registry.DISPLAY[model]
    tbl = prediction_table(panel, lv, name, seed=seed, version=git_commit())
    preds_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_parquet(preds_dir / f"{name}_h{h}.parquet", index=False)
    meta = {"model": name, "horizon": h, "selected_by": "min qlike_val (검증 구간)", "config_id": cid,
            "params": json.loads(best["params"]), "smearing": bool(best["smearing"]), "smearing_c": c,
            "val": {k: best[k] for k in ["qlike_val", "mse_val", "qlike_event", "n_val", "n_event"]},
            "info": info, "seed": seed, "git_commit": git_commit(), "data_dir": str(data_dir),
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                         **{p: sys.modules[p].__version__ for p in ("lightgbm", "torch", "scipy") if p in sys.modules}},
            "reportable": bool(prep_meta.get("reportable", False)), "hold_reasons": prep_meta.get("hold_reasons")}
    (preds_dir / f"{name}_h{h}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                                      encoding="utf-8")
    print(table.head(5).to_string(index=False))
    print(f"[tune] 선택: {cid} smearing={bool(best['smearing'])} → {preds_dir / name}_h{h}.parquet")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.tune")
    ap.add_argument("--model", choices=sorted(registry.DISPLAY), required=True)
    ap.add_argument("--h", type=int, choices=[5, 22], required=True)
    ap.add_argument("--grid", default=str(REPO / "configs" / "tuning.yaml"))
    ap.add_argument("--data-dir", default="data/dev")
    ap.add_argument("--out", default="data/tuning")
    ap.add_argument("--preds", default="data/preds/tuned")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    with open(args.grid, encoding="utf-8") as f:
        grid = (yaml.safe_load(f) or {}).get(args.model) or {}
    tune(args.model, args.h, REPO / args.data_dir, grid, REPO / args.out, REPO / args.preds, args.seed, args.jobs)


if __name__ == "__main__":
    main()
