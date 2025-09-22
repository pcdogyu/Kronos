# run_kronos_demo.py
# 作用：
# 1) 生成一份假的 5 分钟 K 线 CSV（OHLCV+amount），保存在 examples/data/demo.csv
# 2) 加载 Kronos 的 tokenizer + model
# 3) 用 KronosPredictor 做预测，存到 pred_demo.csv 并打印前几行

import os
import sys
import math
import random
import pathlib
import datetime as dt

import pandas as pd
import numpy as np
import torch

# 确保脚本无论从哪里执行，都能 import 到项目里的 model/
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from model import Kronos, KronosTokenizer, KronosPredictor  # 注意：不是 import kronos

# -----------------------------
# 0) 生成一份假的 K 线数据
# -----------------------------
data_dir = ROOT / "examples" / "data"
data_dir.mkdir(parents=True, exist_ok=True)
csv_path = data_dir / "demo.csv"

def make_fake_ohlcv(n=800, start=100.0, seed=42, freq_minutes=5):
    rng = np.random.default_rng(seed)
    # 模拟价格随机游走 + 波动聚集
    rets = rng.normal(loc=0.0002, scale=0.004, size=n)  # 微小上漂
    price = np.empty(n)
    price[0] = start
    for i in range(1, n):
        price[i] = price[i-1] * math.exp(rets[i])

    # 构造 OHLC：用价格 + 小抖动
    base = price
    high = base * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = base * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.r_[base[0], base[:-1]]
    close = base

    # 成交量/额：与波动相关
    vol = (np.abs(rets) * 1e6 * (1 + rng.normal(0, 0.2, n))).clip(min=1000).astype(np.int64)
    amount = (vol * base * (1 + rng.normal(0, 0.05, n))).clip(min=vol*base*0.5)

    start_ts = dt.datetime.now().replace(second=0, microsecond=0) - dt.timedelta(minutes=freq_minutes*(n-1))
    ts = [start_ts + dt.timedelta(minutes=freq_minutes*i) for i in range(n)]

    df = pd.DataFrame({
        "timestamps": ts,
        "open": open_,
        "high": np.maximum.reduce([open_, high, close]),
        "low": np.minimum.reduce([open_, low, close]),
        "close": close,
        "volume": vol,
        "amount": amount
    })
    return df

if not csv_path.exists():
    df_fake = make_fake_ohlcv(n=800, start=100.0, seed=47, freq_minutes=5)
    df_fake.to_csv(csv_path.as_posix(), index=False)
    print(f"[OK] wrote fake data -> {csv_path}")
else:
    print(f"[SKIP] fake data exists -> {csv_path}")

# -----------------------------
# 1) 加载预训练 tokenizer / model
# -----------------------------
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_NAME     = "NeoQuasar/Kronos-small"

print(f"[Info] loading tokenizer: {TOKENIZER_NAME}")
tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)

print(f"[Info] loading model: {MODEL_NAME}")
model = Kronos.from_pretrained(MODEL_NAME)

# -----------------------------
# 2) 构建预测器（自动探测设备）
# -----------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
max_ctx = 512  # small/base 的推荐最大上下文
print(f"[Info] device={device}, max_context={max_ctx}")
predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_ctx)

# -----------------------------
# 3) 读取数据并切分窗口
# -----------------------------
df = pd.read_csv(csv_path)
df["timestamps"] = pd.to_datetime(df["timestamps"])

# 设定历史窗口长度和预测长度
lookback = 400
pred_len = 120

# 保证长度充足
assert len(df) >= lookback + pred_len, "数据太短，调小 lookback/pred_len 或生成更多数据"

use_cols = ['open','high','low','close','volume','amount']  # 你也可以只用 OHLC
x_df = df.loc[:lookback-1, use_cols]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# -----------------------------
# 4) 预测
# -----------------------------
print("[Info] predicting...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1
)

out_path = ROOT / "pred_demo.csv"
pred_df.to_csv(out_path.as_posix(), index=False)
print(f"[OK] saved predictions -> {out_path}")
print(pred_df.head(10))
