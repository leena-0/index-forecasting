"""검증 구간 하이퍼파라미터 튜닝 (CLAUDE.md 4.1: 튜닝·선택은 검증 구간에서 끝내고 테스트는 한 번만 본다).

설정 하나마다
  1. 학습 구간으로 학습하고 모든 행을 예측한다 (registry.fit_predict)
  2. 편향 보정 없음 / 있음(smearing, 학습 구간 잔차로 상수 추정) 두 가지를 모두 평가한다 — 다시 학습하지 않는다
  3. 검증 구간 지표 (분산 단위, CLAUDE.md 7.1):
       qlike_val, mse_val   검증 구간 평균 손실
       qlike_event          검증 구간 편입 이벤트 윈도우 안의 평균 QLIKE (7.4와 같은 범위)
     평가 행 = 검증 구간, embargo == False, yz_var > 0, 그리고 이 모델의 모든 설정이 예측한 행 (설정마다 표본이 같도록)
선택: qlike_val이 가장 작은 (설정, 편향 보정) 조합. 테스트 구간 값은 계산하지 않는다.
"""
import itertools
import json

import numpy as np
import pandas as pd

from src.evaluate.losses import mse, qlike
from src.evaluate.sensitivity import event_window_rows
from src.models import registry
from src.pipeline.features import load_panel

_PANEL = None


def expand(grid):
    """{"a": [1, 2], "b": [3]} → [{"a": 1, "b": 3}, {"a": 2, "b": 3}]. 빈 그리드는 기본값 하나."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def config_id(i, params):
    return f"c{i:02d}_" + "_".join(f"{k}-{v}" for k, v in params.items()) if params else f"c{i:02d}_default"


def _init(data_dir, h):
    global _PANEL
    _PANEL = load_panel(data_dir, h)


def _work(args):
    model, cid, params, seed = args
    pred, info = registry.fit_predict(model, _PANEL, params, seed)
    c = registry.smearing_constant(_PANEL, pred)
    return cid, pred.to_numpy(dtype=np.float32), c, {k: v for k, v in info.items() if k != "train_loss_by_epoch"}


def run_configs(model, data_dir, h, configs, seed=0, jobs=1):
    """설정마다 (config_id, 예측 log_var 배열, 편향 보정 상수, info). jobs > 1이면 프로세스 병렬."""
    tasks = [(model, config_id(i, p), p, seed) for i, p in enumerate(configs)]
    if jobs <= 1:
        _init(data_dir, h)
        return [_work(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(data_dir, h)) as ex:
        return list(ex.map(_work, tasks))


def score(panel, phases, results, configs):
    """설정 × 편향 보정별 검증 구간 지표 표와 평가 행 마스크."""
    base = ((panel["split"] == "val") & ~panel["embargo"] & (panel["yz_var"] > 0)).to_numpy().copy()
    for _, pred, _, _ in results:
        base &= np.isfinite(pred)
    in_event = event_window_rows(panel, phases).to_numpy() & base
    y = panel["yz_var"].to_numpy(dtype=float)
    rows = []
    for (cid, pred, c, info), params in zip(results, configs):
        for smearing in (False, True):
            lv = pred.astype(float) + (c if smearing else 0.0)
            vhat = np.exp(lv)
            q, m = qlike(y[base], vhat[base]), mse(y[base], vhat[base])
            rows.append({"config_id": cid, "params": json.dumps(params, ensure_ascii=False), "smearing": smearing,
                         "smearing_c": round(c, 5), "n_val": int(base.sum()), "qlike_val": float(q.mean()),
                         "mse_val": float(m.mean()), "n_event": int(in_event.sum()),
                         "qlike_event": float(qlike(y[in_event], vhat[in_event]).mean()) if in_event.any() else np.nan})
    table = pd.DataFrame(rows).sort_values("qlike_val", ignore_index=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table, base
