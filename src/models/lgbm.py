"""LightGBM — t일 특성 → 다음 h일 로그 분산 (CLAUDE.md 6.1).

    입력 x_t = [ln rv_d, ln rv_w, ln rv_m, ln yz_5, ln yz_22, ret, log_turnover, log_mcap, is_member, n_obs]_t
    타깃     = log_var_t = ln(구간 t+1 ~ t+h의 Yang–Zhang 분산)
    손실     = L2 (로그 분산의 제곱오차, CLAUDE.md 5.2 "로그 변환한 분산으로 학습")

- 학습: split == train, embargo == False, log_var가 있는 행. 검증·테스트 구간은 쓰지 않는다.
- 분산 특성은 로그로 바꾸고, 0 이하·결측은 NaN으로 둔다 — LightGBM이 결측을 따로 분기한다 (보간 없음).
- 하이퍼파라미터는 고정(PARAMS). 튜닝하지 않았고, 바꾸면 meta.json의 params로 구분한다.
"""
import numpy as np
import pandas as pd

LOG_FEATURES = ["rv_d", "rv_w", "rv_m", "yz_5", "yz_22"]
RAW_FEATURES = ["ret", "log_turnover", "log_mcap", "is_member", "n_obs"]
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
          "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "num_boost_round": 500,
          "deterministic": True, "force_row_wise": True, "verbose": -1}


def design(panel):
    """특성 행렬 (DataFrame). 분산 특성은 로그, 0 이하는 NaN."""
    x = pd.DataFrame(index=panel.index)
    for c in LOG_FEATURES:
        x[f"ln_{c}"] = np.log(panel[c].where(panel[c] > 0))
    for c in RAW_FEATURES:
        x[c] = panel[c].astype(float)
    return x


def fit(panel, seed=0, params=None):
    """학습 구간 행으로 학습. 돌려주는 것: (모델, 학습 행 수)."""
    import lightgbm as lgb

    tr = panel[(panel["split"] == "train") & ~panel["embargo"] & panel["log_var"].notna()]
    p = {**PARAMS, **(params or {}), "seed": seed}
    rounds = p.pop("num_boost_round")
    model = lgb.train(p, lgb.Dataset(design(tr), tr["log_var"]), num_boost_round=rounds)
    return model, len(tr)


def predict(panel, model):
    """예측 로그 분산 Series (panel과 같은 index)."""
    return pd.Series(model.predict(design(panel)), index=panel.index)
