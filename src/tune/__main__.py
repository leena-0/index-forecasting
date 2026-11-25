"""검증 구간 하이퍼파라미터 튜닝 (docs/TUNING.md).

    python -m src.tune --model lgbm --h 5 [--window 2020] [--grid configs/tuning.yaml] [--data-dir data/dev] [--jobs 4]

--window(테스트 연도)를 주면 그 슬라이딩 윈도우의 학습·검증 구간으로 튜닝하고 (CLAUDE.md 4.1), 결과 폴더 밑에
w{테스트 연도}/를 만들어 저장한다. 예측 테이블에는 검증·테스트·다음 해(next) 행만 넣는다.
--event-features를 주면 이벤트 단위 변수를 입력에 더해 튜닝한다 (CLAUDE.md 6.2). 튜닝 폴더는 {model}_ev_h{h}, 예측의 모델
이름은 {모델}+EV다. 그리드는 변수가 없을 때와 같다.

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
from src.pipeline.windows import TEST_YEARS

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


def tune(model, h, data_dir, grid, out_root, preds_dir, seed=0, jobs=1, window=None,
         event_features=False):
    if event_features and model not in registry.EVENT_MODELS:
        raise ValueError(f"이벤트 단위 변수를 받지 않는 모델입니다: {model}")
    configs = expand(grid)
    if window is not None:
        out_root, preds_dir = out_root / f"w{window}", preds_dir / f"w{window}"
    print(f"[tune] {model} h={h} window={window} event_features={event_features}: 설정 {len(configs)}개 × 편향 보정 2가지, jobs={jobs}")
    results = run_configs(model, data_dir, h, configs, seed, jobs, window, event_features)
    panel = load_panel(data_dir, h, window, event_features)
    phases = pd.read_parquet(data_dir / "event_phases.parquet") if window is None else None
    table, mask = score(panel, phases, results, configs)

    out = out_root / f"{model}{'_ev' if event_features else ''}_h{h}"
    (out / "val_preds").mkdir(parents=True, exist_ok=True)
    keys = panel.loc[mask, ["ticker", "origin_date"]].reset_index(drop=True)
    for cid, pred, c, _ in results:
        keys.assign(pred_log_var=pred[mask], pred_log_var_smeared=pred[mask] + c).to_parquet(
            out / "val_preds" / f"{cid}.parquet", index=False)
    prep_meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    split = prep_meta.get("config", {}).get("split") if window is None else f"슬라이딩 윈도우, 테스트 {window}"
    header = (f"# {registry.DISPLAY[model]} h={h} 튜닝 (검증 구간, {split})\n\n"
              f"개발용 — reportable={prep_meta.get('reportable')}, hold_reasons={prep_meta.get('hold_reasons')}\n\n")
    table.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(header + to_markdown(table), encoding="utf-8")

    best = table.iloc[0]
    i = [r[0] for r in results].index(best["config_id"])
    cid, pred, c, info = results[i]
    lv = pred.astype(float) + (c if best["smearing"] else 0.0)
    name = registry.DISPLAY[model] + (registry.EVENT_SUFFIX if event_features else "")
    tbl = prediction_table(panel, lv, name, seed=seed, version=git_commit())
    if window is not None:  # 학습 구간과 그 전의 예측은 평가에 쓰지 않는다
        tbl = tbl[tbl["origin_date"].dt.year >= window - 1].reset_index(drop=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_parquet(preds_dir / f"{name}_h{h}.parquet", index=False)
    meta = {"model": name, "horizon": h, "window": window, "event_features": event_features, "selected_by": "min qlike_val (검증 구간)", "config_id": cid,
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
    ap.add_argument("--window", type=int, choices=TEST_YEARS, default=None, help="슬라이딩 윈도우의 테스트 연도")
    ap.add_argument("--event-features", action="store_true", help="이벤트 단위 변수를 입력에 더함")
    ap.add_argument("--grid", default=str(REPO / "configs" / "tuning.yaml"))
    ap.add_argument("--data-dir", default="data/dev")
    ap.add_argument("--out", default="data/tuning")
    ap.add_argument("--preds", default="data/preds/tuned")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    with open(args.grid, encoding="utf-8") as f:
        grid = (yaml.safe_load(f) or {}).get(args.model) or {}
    tune(args.model, args.h, REPO / args.data_dir, grid, REPO / args.out, REPO / args.preds, args.seed, args.jobs,
         args.window, args.event_features)


if __name__ == "__main__":
    main()
