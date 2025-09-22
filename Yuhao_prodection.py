# Yuhao_prodection.py  —  UTC+8、预测从“现在整点”起共48h（先锚定到最后真实价，再向前预测）、背景从现在起高亮、网格+左右Y轴
# 用法：
#   1) python Yuhao_prodection.py BTCUSDT 7d
#      历史：北京时间过去第八天00:00->昨天24:00 + 今天00:00->现在整点（蓝）
#      预测：从“现在整点”起共48h；预测前先将“现在整点”的预测价锚定到最后真实收盘
#   2) python Yuhao_prodection.py BTCUSDT
#      历史：现在整点往回168h（蓝）；预测：同样从“现在整点”起共48h；锚定方式一致
# 输出：examples/data/<SYMBOL>_YYYYMMDD-HHMM.csv/.png（文件名用UTC+8）

import sys, pathlib, datetime as dt
import requests
import pandas as pd
from zoneinfo import ZoneInfo

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from model import Kronos, KronosTokenizer, KronosPredictor

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
TZ = ZoneInfo("Asia/Shanghai")  # UTC+8

# ---------- 时间工具 ----------
def now_cst_floor_min():
    n = dt.datetime.now(TZ)
    return n.replace(second=0, microsecond=0)

def now_cst_floor_hour():
    n = dt.datetime.now(TZ)
    return n.replace(minute=0, second=0, microsecond=0)

def cst_midnight(t: dt.datetime) -> dt.datetime:
    return t.astimezone(TZ).replace(hour=0, minute=0, second=0, microsecond=0)

def to_ms_utc(ts: dt.datetime) -> int:
    # Binance 的 startTime/endTime 永远按 UTC 解释：先转 UTC 再取毫秒
    return int(ts.astimezone(dt.timezone.utc).timestamp() * 1000)

# ---------- 拉取 1h K线（输入/输出均按 UTC+8） ----------
def fetch_klines_1h_cst(symbol: str, start_cst: dt.datetime, end_cst: dt.datetime) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": "1h",
        "startTime": to_ms_utc(start_cst),
        "endTime": to_ms_utc(end_cst),
        "limit": 1000,
    }
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for k in data:
        ts_utc = pd.to_datetime(int(k[0]), unit="ms", utc=True)
        ts_cst = ts_utc.tz_convert(TZ)
        rows.append({
            "timestamps": ts_cst,
            "open":  float(k[1]),
            "high":  float(k[2]),
            "low":   float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "amount": float(k[7]),
        })
    return pd.DataFrame(rows).sort_values("timestamps").reset_index(drop=True)

# ---------- 参数 ----------
def parse_args():
    if len(sys.argv) not in (2, 3):
        print("用法:\n  python Yuhao_prodection.py <SYMBOL> [7d]\n示例:\n  python Yuhao_prodection.py BTCUSDT 7d\n  python Yuhao_prodection.py BTCUSDT")
        sys.exit(1)
    symbol = sys.argv[1].upper()
    window = sys.argv[2].lower() if len(sys.argv) == 3 else None
    if window is not None and window != "7d":
        print("第二参数仅支持 7d，或省略。")
        sys.exit(1)
    return symbol, window

# ---------- 主流程 ----------
def main():
    symbol, window = parse_args()

    now_h = now_cst_floor_hour()
    pred_start_cst = now_h
    pred_periods = 48

    if window == "7d":
        today_00 = cst_midnight(now_h)
        start_cst = today_00 - dt.timedelta(days=7)
        end_cst   = today_00                # 历史到昨天24:00
        expected_hist = 7 * 24
    else:
        end_cst = now_h
        start_cst = end_cst - dt.timedelta(hours=168)
        expected_hist = 168

    # 1) 历史（到昨天24:00）
    hist_df = fetch_klines_1h_cst(symbol, start_cst, end_cst)
    if hist_df.empty:
        raise RuntimeError("历史行情为空。")
    if len(hist_df) > expected_hist:
        hist_df = hist_df.tail(expected_hist).reset_index(drop=True)

    # 2) 今日真实（7d模式补齐 今天00:00 -> 现在整点）
    today_real = pd.DataFrame()
    if window == "7d":
        today_00 = cst_midnight(now_h)
        if now_h > today_00:
            today_real = fetch_klines_1h_cst(symbol, today_00, now_h)

    # 给“蓝线”绘图用的数据：历史 + 今日真实（如有），并按时间排序
    hist_plot_df = hist_df if today_real.empty else pd.concat([hist_df, today_real], ignore_index=True)
    hist_plot_df = hist_plot_df.sort_values("timestamps").reset_index(drop=True)

    # 若真实数据尚未覆盖到预测起点，则复制最后一根并补齐到预测起点，方便在图上衔接
    if not hist_plot_df.empty and hist_plot_df["timestamps"].iloc[-1] < pred_start_cst:
        pad_row = hist_plot_df.tail(1).copy()
        pad_row.loc[:, "timestamps"] = pred_start_cst
        hist_plot_df = pd.concat([hist_plot_df, pad_row], ignore_index=True)

    # 让模型只看到预测起点之前的真实数据，避免泄漏预测区间的真实价格
    history_for_model = hist_plot_df[hist_plot_df["timestamps"] < pred_start_cst].copy()
    if history_for_model.empty:
        raise RuntimeError("历史数据不足以覆盖预测起点之前的窗口，请增加 fetch 时长。")

    # 3) 预测时间戳（Series, tz=UTC+8）
    y_ts = pd.Series(pd.date_range(start=pred_start_cst, periods=pred_periods, freq="1h", tz=TZ))

    # 4) Kronos 预测
    tok_name = "NeoQuasar/Kronos-Tokenizer-base"
    mdl_name = "NeoQuasar/Kronos-small"
    tokenizer = KronosTokenizer.from_pretrained(tok_name)
    model = Kronos.from_pretrained(mdl_name)

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    use_cols = ["open", "high", "low", "close", "volume", "amount"]
    x_df = history_for_model[use_cols]
    x_ts = history_for_model["timestamps"]
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts,
        y_timestamp=y_ts, pred_len=len(y_ts),
        T=1.0, top_p=0.9, sample_count=1
    )

    # 5) === 关键修复：在“现在整点”做重锚定 ===
    pred_out = pred_df.reset_index().rename(columns={"index": "timestamps"})
    pred_out["timestamps"] = pd.to_datetime(pred_out["timestamps"]).dt.tz_convert(TZ)

    # 锚价 = 最后一根真实K线的收盘（优先：今日真实的最后一根；否则：历史最后一根）
    if not today_real.empty:
        anchor_price = float(today_real["close"].iloc[-1])
    else:
        anchor_price = float(hist_plot_df["close"].iloc[-1])

    # 找到“现在整点”对应的预测行（严格匹配；若无，则用最近一行兜底）
    mask_now = pred_out["timestamps"] == now_h
    if mask_now.any():
        idx_now = pred_out.index[mask_now][0]
    else:
        idx_now = (pred_out["timestamps"] - now_h).abs().idxmin()

    current_pred_close = float(pred_out.at[idx_now, "close"])
    delta = anchor_price - current_pred_close
    for c in ["open", "high", "low", "close"]:
        pred_out[c] = pred_out[c] + delta
    pred_out.loc[idx_now, ["open", "high", "low", "close"]] = anchor_price

    # 6) 合并输出 CSV（timestamps=UTC+8）
    pred_out["is_pred"] = 1
    hist_out = hist_plot_df.copy(); hist_out["is_pred"] = 0
    combined = pd.concat([hist_out, pred_out], ignore_index=True)
    combined = combined[["timestamps","open","high","low","close","volume","amount","is_pred"]]

    stamp = now_cst_floor_min().strftime("%Y%m%d-%H%M")
    out_dir = ROOT / "examples" / "data"; out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{symbol}_{stamp}.csv"
    combined.to_csv(out_csv, index=False)
    print(f"[OK] CSV -> {out_csv}  |  rebase(now) Δ = {delta:.6f}")

    # 7) 绘图：蓝=真实历史，绿=预测（未来48小时）；背景从现在起高亮；网格+双Y轴
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Microsoft JhengHei","Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        hist_plot_df["timestamps"],
        hist_plot_df["close"],
        label="历史收盘（真实）",
        linewidth=1.6,
        color="tab:blue"
    )

    pred_future = pred_out[pred_out["timestamps"] >= pred_start_cst]
    if not pred_future.empty:
        ax.plot(
            pred_future["timestamps"],
            pred_future["close"],
            label="预测收盘（未来48h）",
            linewidth=1.8,
            color="tab:green"
        )
        ax.axvspan(pred_start_cst, pred_future["timestamps"].iloc[-1], alpha=0.12, color="tab:green")

    ax.set_axisbelow(True); ax.minorticks_on()
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.25)

    ax_r = ax.twinx()
    ax_r.set_ylabel("收盘价（右）")
    ax_r.set_ylim(ax.get_ylim())
    ax_r.yaxis.set_major_locator(ax.yaxis.get_major_locator())
    ax_r.yaxis.set_major_formatter(ax.yaxis.get_major_formatter())
    ax_r.grid(False)

    ax.set_xlabel("时间（UTC+8）"); ax.set_ylabel("收盘价（左）")
    title_suffix = (
        "7日历史 + 预测（未来48小时）"
        if window == "7d"
        else "近168小时历史 + 预测（未来48小时）"
    )
    ax.set_title(f"{symbol} —— {title_suffix}")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate(); ax.legend(); fig.tight_layout()

    out_png = out_dir / f"{symbol}_{stamp}.png"
    fig.savefig(out_png, dpi=150)
    print(f"[OK] PNG -> {out_png}")

    try:
        plt.show()
    except KeyboardInterrupt:
        print("[WARN] 图形显示被中断；PNG 已保存。")
    except Exception:
        pass
    finally:
        plt.close(fig)

if __name__ == "__main__":
    main()
