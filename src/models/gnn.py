"""GNN — 날짜별 종목 그래프에서 다음 h일 로그 분산 (CLAUDE.md 6.1 2단계, 6.3 설계 제안).

그래프 (t일)
- 노드: t일 특성이 모두 있는 universe 종목 (한 번이라도 구성종목이었던 종목 = 구성종목 + 편입·편출 이벤트 종목).
  편입 전 종목도 노드라 편입 전 구간도 예측한다 (6.3).
- 엣지: refresh일마다 최근 window거래일 종가 수익률 상관(관측 20일 이상)을 계산하고, t일에는 t 이전 가장 최근
  refresh일의 상관을 쓴다. 각 노드는 그날 노드 중 상관이 큰 상위 k개 이웃과 연결 (이웃이 없으면 자기 특성만).
- 노드 특성 x = [ln rv_d, ln rv_w, ln rv_m, ret, log_turnover, log_mcap, is_member] (학습 구간 통계로 표준화)

모델: GraphSAGE 평균 집계 2층 (Hamilton et al., 2017)
    h¹ = ReLU(W₁ x_i + U₁ mean_{j∈N(i)} x_j),  h² = ReLU(W₂ h¹_i + U₂ mean_{j∈N(i)} h¹_j),  ŷ_i = v·h²_i
손실: MSE (로그 분산). 학습: 학습 구간 날짜만, 에포크 고정. t일 그래프·특성은 t일까지의 정보만 쓴다.
"""
import numpy as np
import pandas as pd

NODE_FEATURES = ["rv_d", "rv_w", "rv_m", "ret", "log_turnover", "log_mcap", "is_member"]
LOG_FEATURES = {"rv_d", "rv_w", "rv_m"}
PARAMS = {"k": 5, "window": 60, "refresh": 5, "hidden": 32, "epochs": 5, "lr": 1e-3}


def node_features(panel):
    cols = []
    for c in NODE_FEATURES:
        v = panel[c].astype(float)
        cols.append(np.log(v.where(v > 0)) if c in LOG_FEATURES else v)
    return np.column_stack(cols).astype(np.float32)


def build_graphs(panel, k, window, refresh):
    """날짜별 (행 위치 배열, 이웃 위치 배열[n × k], 이웃 마스크[n × k]). 위치는 그날 노드 안의 순번."""
    X = node_features(panel)
    valid = np.isfinite(X).all(axis=1)
    dates = np.sort(panel["origin_date"].unique())
    tickers = np.sort(panel["ticker"].unique())
    R = panel.pivot_table(index="origin_date", columns="ticker", values="ret").reindex(index=dates, columns=tickers)
    t_pos = {t: i for i, t in enumerate(tickers)}
    by_date = panel.assign(_row=np.arange(len(panel)), _valid=valid).groupby("origin_date")
    graphs, corr = {}, None
    for di, d in enumerate(dates):
        if di % refresh == 0:  # refresh일의 상관 (d일까지의 수익률)
            corr = R.iloc[max(0, di - window + 1): di + 1].corr(min_periods=20).to_numpy()
        g = by_date.get_group(d)
        g = g[g["_valid"]]
        if g.empty:
            continue
        rows = g["_row"].to_numpy()
        ti = np.array([t_pos[t] for t in g["ticker"]])
        c = corr[np.ix_(ti, ti)].copy()
        np.fill_diagonal(c, np.nan)
        c = np.where(np.isfinite(c), c, -np.inf)
        kk = min(k, len(rows) - 1)
        if kk <= 0:
            nb, mask = np.zeros((len(rows), 1), int), np.zeros((len(rows), 1), bool)
        else:
            nb = np.argpartition(-c, kk - 1, axis=1)[:, :kk]
            mask = np.isfinite(np.take_along_axis(c, nb, axis=1))
        graphs[d] = (rows, nb, mask)
    return graphs, X


def _net(n_in, hidden):
    import torch

    class Sage(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.self_lin, self.nb_lin = torch.nn.Linear(a, b), torch.nn.Linear(a, b, bias=False)

        def forward(self, x, nb, mask):
            m = mask.float().unsqueeze(-1)
            agg = (x[nb] * m).sum(1) / m.sum(1).clamp(min=1.0)
            return torch.relu(self.self_lin(x) + self.nb_lin(agg))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1, self.l2 = Sage(n_in, hidden), Sage(hidden, hidden)
            self.head = torch.nn.Linear(hidden, 1)

        def forward(self, x, nb, mask):
            return self.head(self.l2(self.l1(x, nb, mask), nb, mask)).squeeze(-1)

    return Net()


def fit_predict(panel, seed=0, params=None):
    """학습 구간 날짜로 학습한 뒤 모든 날짜를 예측. 돌려주는 것: (예측 log_var Series, info)."""
    import torch

    p = {**PARAMS, **(params or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    graphs, X = build_graphs(panel, p["k"], p["window"], p["refresh"])
    split = panel["split"].to_numpy()
    usable = ~panel["embargo"].to_numpy() & np.isfinite(panel["log_var"].to_numpy(dtype=float))
    y = panel["log_var"].to_numpy(dtype=np.float32)
    train_dates = [d for d, (rows, _, _) in graphs.items() if split[rows[0]] == "train"]
    tr_rows = np.concatenate([graphs[d][0] for d in train_dates])
    mu, sd = X[tr_rows].mean(axis=0), X[tr_rows].std(axis=0) + 1e-8
    Z = (X - mu) / sd
    tr_y = tr_rows[usable[tr_rows]]
    y_mu, y_sd = y[tr_y].mean(), y[tr_y].std()

    def tensors(d):
        rows, nb, mask = graphs[d]
        return (torch.from_numpy(Z[rows]), torch.from_numpy(nb), torch.from_numpy(mask), rows)

    net = _net(Z.shape[1], p["hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(p["epochs"]):
        net.train()
        total, n = 0.0, 0
        for d in rng.permutation(train_dates):
            x, nb, mask, rows = tensors(d)
            ok = usable[rows]
            if not ok.any():
                continue
            target = torch.from_numpy((y[rows][ok] - y_mu) / y_sd)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x, nb, mask)[torch.from_numpy(ok)], target)
            loss.backward()
            opt.step()
            total += loss.item() * ok.sum()
            n += ok.sum()
        losses.append(round(total / max(n, 1), 5))

    net.eval()
    out = np.full(len(panel), np.nan)
    with torch.no_grad():
        for d in graphs:
            x, nb, mask, rows = tensors(d)
            out[rows] = net(x, nb, mask).numpy() * y_sd + y_mu
    info = {**p, "n_train_dates": len(train_dates), "train_loss_by_epoch": losses, "node_features": NODE_FEATURES}
    return pd.Series(out, index=panel.index), info
