"""모델 등록부: 모든 모델을 같은 방식으로 부른다 (src.models CLI, src.tune).

    fit_predict(model, panel, params, seed) → (pred_log_var: Series, info: dict)

params는 모델 기본값(DEFAULTS)에 덮어쓴다. 공통 선택지 smearing(기본 False)이 True면 편향 보정을 더한다:

    c = ln( mean_i exp(y_i − ŷ_i) ),  i = 학습 구간 행 (split == train, embargo == False, y·ŷ 유한)
    보정 예측 = ŷ + c

로그 분산으로 예측하면 exp(ŷ)는 조건부 평균이 아니라 중앙값에 가깝다(Jensen). c는 그 차이를 학습 구간 잔차로
추정한 상수다 (Duan, 1983, smearing estimate). 검증·테스트 구간 정보는 쓰지 않는다.
"""
import numpy as np

from . import ewma, har, lgbm, lstm

DISPLAY = {"ewma": "EWMA", "har": "HAR", "lgbm": "LightGBM", "lstm": "LSTM"}
DEFAULTS = {"ewma": {"lam": 0.94, "warmup": 22}, "har": {}, "lgbm": dict(lgbm.PARAMS), "lstm": dict(lstm.PARAMS)}


def smearing_constant(panel, pred):
    """학습 구간 잔차로 c = ln mean exp(y − ŷ)."""
    tr = (panel["split"] == "train").to_numpy() & ~panel["embargo"].to_numpy()
    y, p = panel["log_var"].to_numpy(dtype=float), np.asarray(pred, dtype=float)
    ok = tr & np.isfinite(y) & np.isfinite(p)
    return float(np.log(np.mean(np.exp(y[ok] - p[ok]))))


def fit_predict(model, panel, params=None, seed=0):
    if model not in DISPLAY:
        raise ValueError(f"지원하지 않는 모델: {model}")
    p = {**DEFAULTS[model], **(params or {})}
    smearing = bool(p.pop("smearing", False))
    if model == "ewma":
        var = ewma.predict(panel, p["lam"], p["warmup"])
        pred, info = np.log(var.where(var > 0)), dict(p)
    elif model == "har":
        beta, n_fit = har.fit(panel)
        pred = har.predict(panel, beta)
        info = {"beta": dict(zip(["const"] + har.REGRESSORS, beta.tolist())), "n_fit": n_fit}
    elif model == "lgbm":
        fitted, n_fit = lgbm.fit(panel, seed, p)
        pred, info = lgbm.predict(panel, fitted), {**p, "n_fit": n_fit}
    else:
        pred, info = lstm.fit_predict(panel, seed, p)
    info["smearing"] = smearing
    if smearing:
        info["smearing_c"] = smearing_constant(panel, pred)
        pred = pred + info["smearing_c"]
    return pred, info
