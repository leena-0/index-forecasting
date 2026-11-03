"""시퀀스 모델 (LSTM, GRU, TCN, Transformer) — 최근 L거래일의 일별 특성 시퀀스 → 다음 h일 로그 분산 (CLAUDE.md 6.1).

    입력 X_t = [z_{t−L+1}, …, z_t],  z_s = [ln rv_d, ret, log_turnover]_s  (학습 구간 평균·표준편차로 표준화)
    출력     = Linear(마지막 시점의 은닉 표현) = 예측 log_var_t
    arch     lstm (1단계) · gru (2단계): 순환 셀, hidden = 은닉 크기, layers = 층 수
             tcn (3단계): 인과 팽창 합성곱 (Bai et al., 2018) — 층 i의 팽창 2^i, kernel 크기, 잔차 연결, hidden = 채널
             transformer (3단계): 선형 투영 + 학습 위치 임베딩 + 인코더 (Vaswani et al., 2017), hidden = d_model,
                          heads = 헤드 수, 마스크 없음(창 전체가 t일 이전 정보)
    손실     = MSE (로그 분산, CLAUDE.md 5.2)

- 시퀀스는 종목별로 달력상 연속한 L일이다 (특성 패널은 종목마다 첫 행 ~ 마지막 행을 달력에 맞춰 펼친 표).
  창 안에 결측(거래정지, 상장 직후, rv_d ≤ 0)이 하나라도 있으면 그 날은 예측하지 않는다 (보간 없음).
- t일 입력은 t일까지의 특성만 쓴다 → 시점 누수 없음 (tests/test_models.py).
- 학습: split == train, embargo == False, log_var가 있는 행. 에포크 수 고정(조기 종료 없음 — 검증 구간을 학습에 쓰지 않는다).
- 재현: torch.manual_seed(seed), numpy 난수, 단일 스레드 결정적 연산.
"""
import numpy as np
import pandas as pd

SEQ_FEATURES = ["rv_d", "ret", "log_turnover"]
PARAMS = {"arch": "lstm", "lookback": 22, "hidden": 32, "layers": 1, "epochs": 5, "batch_size": 1024, "lr": 1e-3,
          "kernel": 3, "heads": 2}


def daily_features(panel):
    """일별 특성 행렬 (n × 3). rv_d는 로그, 0 이하는 NaN."""
    rv = panel["rv_d"].where(panel["rv_d"] > 0)
    return np.column_stack([np.log(rv), panel["ret"], panel["log_turnover"]]).astype(np.float32)


def sequences(panel, lookback):
    """종목별 L일 창. 돌려주는 것: (X[m × L × 3], 행 위치[m]) — 창 안에 결측이 없는 행만.
    panel은 종목·날짜 순으로 정렬되어 있어야 한다 (load_panel 결과를 sort_values(["ticker", "origin_date"]))."""
    z = daily_features(panel)
    tickers = panel["ticker"].to_numpy()
    xs, rows = [], []
    start = 0
    for i in range(1, len(panel) + 1):
        if i < len(panel) and tickers[i] == tickers[start]:
            continue
        g = z[start:i]  # 한 종목
        if len(g) >= lookback:
            w = np.lib.stride_tricks.sliding_window_view(g, (lookback, g.shape[1]))[:, 0]  # (n−L+1) × L × 3
            ok = np.isfinite(w).all(axis=(1, 2))
            xs.append(w[ok])
            rows.append(start + lookback - 1 + np.flatnonzero(ok))
        start = i
    if not xs:
        return np.empty((0, lookback, len(SEQ_FEATURES)), np.float32), np.empty(0, int)
    return np.concatenate(xs), np.concatenate(rows)


def _net(n_in, p):
    """arch에 맞는 신경망. 입력 (배치 × L × 특성) → (배치,) 예측 log_var (표준화된 값)."""
    import torch

    rnn = {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU}
    if p["arch"] == "tcn":
        return _tcn(n_in, p)
    if p["arch"] == "transformer":
        return _transformer(n_in, p)
    if p["arch"] not in rnn:
        raise ValueError(f"지원하지 않는 arch: {p['arch']}")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn[p["arch"]](n_in, p["hidden"], num_layers=p["layers"], batch_first=True)
            self.head = torch.nn.Linear(p["hidden"], 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1]).squeeze(-1)

    return Net()


def _tcn(n_in, p):
    import torch

    class Block(torch.nn.Module):
        def __init__(self, c_in, c_out, dilation):
            super().__init__()
            self.pad = (p["kernel"] - 1) * dilation  # 왼쪽만 채워 인과(t 이후를 보지 않음)
            self.conv = torch.nn.Conv1d(c_in, c_out, p["kernel"], dilation=dilation)
            self.skip = torch.nn.Conv1d(c_in, c_out, 1) if c_in != c_out else torch.nn.Identity()

        def forward(self, x):  # x: 배치 × 채널 × L
            y = self.conv(torch.nn.functional.pad(x, (self.pad, 0)))
            return torch.relu(y + self.skip(x))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            chans = [n_in] + [p["hidden"]] * p["layers"]
            self.blocks = torch.nn.Sequential(*[Block(chans[i], chans[i + 1], 2 ** i) for i in range(p["layers"])])
            self.head = torch.nn.Linear(p["hidden"], 1)

        def forward(self, x):  # x: 배치 × L × 특성
            return self.head(self.blocks(x.transpose(1, 2))[:, :, -1]).squeeze(-1)

    return Net()


def _transformer(n_in, p):
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            d = p["hidden"]
            self.proj = torch.nn.Linear(n_in, d)
            self.pos = torch.nn.Parameter(torch.zeros(1, p["lookback"], d))
            layer = torch.nn.TransformerEncoderLayer(d, p["heads"], dim_feedforward=2 * d, dropout=0.0,
                                                     batch_first=True)
            self.enc = torch.nn.TransformerEncoder(layer, p["layers"], enable_nested_tensor=False)
            self.head = torch.nn.Linear(d, 1)

        def forward(self, x):
            return self.head(self.enc(self.proj(x) + self.pos)[:, -1]).squeeze(-1)

    return Net()


def fit_predict(panel, seed=0, params=None):
    """학습 구간으로 학습한 뒤 모든 행을 예측. 돌려주는 것: (예측 log_var Series, 정보 dict)."""
    import torch

    p = {**PARAMS, **(params or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    panel = panel.sort_values(["ticker", "origin_date"])
    X, rows = sequences(panel, p["lookback"])
    y = panel["log_var"].to_numpy(dtype=np.float32)[rows]
    is_tr = ((panel["split"].to_numpy()[rows] == "train") & ~panel["embargo"].to_numpy()[rows] & np.isfinite(y))
    mu = X[is_tr].reshape(-1, X.shape[2]).mean(axis=0)
    sd = X[is_tr].reshape(-1, X.shape[2]).std(axis=0) + 1e-8
    X = (X - mu) / sd
    y_mu, y_sd = y[is_tr].mean(), y[is_tr].std()

    net = _net(X.shape[2], p)
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
    Xt, yt = torch.from_numpy(X[is_tr]), torch.from_numpy((y[is_tr] - y_mu) / y_sd)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(p["epochs"]):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for b in range(0, len(Xt), p["batch_size"]):
            idx = perm[b:b + p["batch_size"]]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(Xt))

    net.eval()
    out = np.empty(len(X), np.float32)
    with torch.no_grad():
        for b in range(0, len(X), 8192):
            out[b:b + 8192] = net(torch.from_numpy(X[b:b + 8192])).numpy()
    pred = pd.Series(np.nan, index=panel.index)
    pred.iloc[rows] = out * y_sd + y_mu
    info = {**p, "n_train": int(is_tr.sum()), "train_loss_by_epoch": [round(v, 5) for v in losses],
            "seq_features": SEQ_FEATURES}
    return pred, info
