# demo_all.py  —  生成假CSV -> 用假数据预测 -> 存 CSV 到指定目录与文件名 -> 可视化
import os, sys, pathlib, datetime as dt
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 确保能 import 到项目内的 model/
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from model import Kronos, KronosTokenizer, KronosPredictor

def make_fake_ohlcv(n=800, start=100.0, seed=123, freq_minutes=5):
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.0002, scale=0.004, size=n)
    price = np.empty(n)
    price[0] = start
    for i in range(1, n):
        price[i] = price[i-1] * np.exp(rets[i])

    base = price
    open_ = np.r_[base[0], base[:-1]]
    high  = np.maximum.reduce([open_, base * (1 + np.abs(rng.normal(0, 0.002, n))), base])
    low   = np.minimum.reduce([open_, base * (1 - np.abs(rng.normal(0, 0.002, n))), base])
    close = base

    vol = (np.abs(rets) * 1e6 * (1 + rng.normal(0, 0.2, n))).clip(min=1000).astype(int)
    amount = (vol * base * (1 + rng.normal(0, 0.05, n))).clip(min=vol*base*0.5)

    start_ts = dt.datetime.now().replace(second=0, microsecond=0) - dt.timedelta(minutes=freq_minutes*(n-1))
    ts = [start_ts + dt.timedelta(minutes=freq_minutes*i) for i in range(n)]

    return pd.DataFrame({
        "timestamps": ts,
        "open": open_,
        "high": high,
        "low":  low,
        "close": close,
        "volume": vol,
        "amount": amount
    })

def main():
    # ---- 1) 原始行情文件生成 ----
    data_dir = ROOT / "examples" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_original = data_dir / "demo_full_production.csv"
    if not csv_original.exists():
        df_fake = make_fake_ohlcv(n=800, start=100.0, seed=123, freq_minutes=5)
        df_fake.to_csv(csv_original, index=False)
        print(f"[OK] 原始行情文件生成 -> {csv_original}")
    else:
        print(f"[SKIP] 原始行情已存在 -> {csv_original}")

    # ---- 2) 加载模型与 tokenizer ----
    tok_name = "NeoQuasar/Kronos-Tokenizer-base"
    mdl_name = "NeoQuasar/Kronos-small"
    print(f"[Info] loading tokenizer: {tok_name}")
    tokenizer = KronosTokenizer.from_pretrained(tok_name)
    print(f"[Info] loading model: {mdl_name}")
    model = Kronos.from_pretrained(mdl_name)

    # ---- 3) 预测器（自动选设备） ----
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_ctx = 512
    print(f"[Info] device={device}, max_context={max_ctx}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_ctx)

    # ---- 4) 读取数据与划分窗口 ----
    df = pd.read_csv(csv_original, parse_dates=["timestamps"])
    lookback, pred_len = 400, 120
    assert len(df) >= lookback + pred_len, "数据太短，调小 lookback/pred_len"

    cols = ["open","high","low","close","volume","amount"]
    x_df = df.loc[:lookback-1, cols]
    x_ts = df.loc[:lookback-1, "timestamps"]
    y_ts = df.loc[lookback:lookback+pred_len-1, "timestamps"]

    # ---- 5) 做预测 ----
    print("[Info] predicting ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1
    )

    # ---- 6) 保存预测后的文件 ----
    pred_dir = ROOT / "examples" / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)
    csv_pred = pred_dir / "demo_full_prediction.csv"
    pred_df.to_csv(csv_pred, index=True)  # 索引是 y_timestamp
    print(f"[OK] 预测结果文件保存 -> {csv_pred}")
    print(pred_df.head(10))

    # ---- 7) 可视化收盘价(close) ----
    x_hist = df["timestamps"][:lookback]
    y_hist = df["close"][:lookback]

    try:
        x_pred = pd.to_datetime(pred_df.index)
    except Exception:
        x_pred = pred_df.index

    y_pred = pred_df["close"]

    plt.figure(figsize=(12, 6))
    plt.plot(x_hist, y_hist, label="history close")
    plt.plot(x_pred, y_pred, label="pred close")
    plt.xlabel("Time"); plt.ylabel("Close")
    plt.title("Kronos Demo — Close Prediction (Production)")
    plt.legend(); plt.tight_layout()

    plot_path = pred_dir / "demo_full_prediction_plot.png"
    plt.savefig(plot_path)
    print(f"[OK] 可视化图保存 -> {plot_path}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
