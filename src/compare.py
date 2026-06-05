# 读取Montel模型预测结果和自己模型的结果，找出共同时间点的交集
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil

def plot_windows_with_time(ground_truth, montel_predict, model_predict, time_array,
                           horizon, k=5, random_seed=42, 
                           save_dir=None, show=True, figsize=(14, 6), rename=None):
    """
    随机选取 k 个窗口，每个窗口单独绘制一张图，横轴为绝对时间，
    仅显示窗口的开头、中间、结尾三个时间点标签。

    Parameters
    ----------
    ground_truth, montel_predict, model_predict : np.ndarray
        一维数组，长度相同。
    time_array : np.ndarray (datetime64 or string)
        时间戳数组，长度与上面相同。
    horizon : int
        窗口长度（如 1056）。
    k : int
        随机选取的窗口数。
    random_seed : int or None
        随机种子。
    time_format : str
        时间显示格式，例如 '%m-%d %H:%M'。
    save_dir : str or None
        保存图片的目录，若 None 则不保存。
    show : bool
        是否显示图片。
    figsize : tuple
        每张图的大小 (width, height)。
    """
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    # 数据校验
    total_len = len(ground_truth)
    n_windows = total_len // horizon
    if total_len % horizon != 0:
        raise ValueError(f"总长度 {total_len} 不是 horizon({horizon}) 的整数倍")
    if montel_predict is not None:
        assert len(montel_predict) == total_len
    assert len(model_predict) == total_len and len(time_array) == total_len

    if random_seed is not None:
        random.seed(random_seed)

    if k > n_windows:
        print(f"警告: k({k}) > 窗口总数({n_windows})，将显示所有窗口")
        indices = list(range(n_windows))
    else:
        indices = random.sample(range(n_windows), k)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for win_idx in indices:
        start = win_idx * horizon
        end = start + horizon
        window_time = time_array[start:end]
        # 确保 time 是 pandas Timestamp 类型，避免 matplotlib 时区警告
        window_time = pd.to_datetime(window_time)
        # 移除时区信息，避免 matplotlib 警告
        window_time = window_time.tz_localize(None)

        # 定位开头、中间、结尾三个时间点
        mid = start + horizon // 2
        # 使用处理后的 window_time 获取 ticks，避免时区警告
        # DatetimeIndex 使用索引访问，不是 iloc
        ticks = [window_time[0], window_time[horizon//2], window_time[-1]]
        tick_labels = [t.strftime('%m-%d %H:%M') for t in ticks]

        plt.figure(figsize=figsize)
        plt.plot(window_time, ground_truth[start:end], label='Actual', color='black', linewidth=1.5)
        if montel_predict is not None:
            plt.plot(window_time, montel_predict[start:end], label='Montel', 
                     linestyle='--', color='blue', linewidth=1)
        if rename is None:
            rename = 'ours'
        plt.plot(window_time, model_predict[start:end], label=rename, 
                 linestyle=':', color='red', linewidth=1.5)
        plt.title(f'from {ticks[0]} to {ticks[-1]}')
        plt.xlabel('time')
        plt.ylabel('price')
        plt.xticks(ticks, tick_labels, fontsize=8)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'window_{win_idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

def mae(y, yhat) -> float:
    """计算平均绝对误差（Mean Absolute Error）"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.abs(y[mask] - yhat[mask]))) if mask.any() else np.nan


def rmse(y, yhat) -> float:
    """计算均方根误差（Root Mean Squared Error）"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(np.mean((y[mask] - yhat[mask]) ** 2))) if mask.any() else np.nan


def bias(y, yhat) -> float:
    """计算偏差（Bias）"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(yhat[mask] - y[mask])) if mask.any() else np.nan


def wape(y, yhat, eps: float = 1e-8) -> float:
    """计算加权绝对百分比误差（Weighted Absolute Percentage Error）"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not mask.any():
        return np.nan
    denom = float(np.sum(np.abs(y[mask])))
    return float(np.sum(np.abs(y[mask] - yhat[mask])) / max(denom, eps))

with open('config.yaml') as f:
    config = yaml.safe_load(f)


# 1. 读取 Montel 预测结果
montel_pred = pd.read_csv(config['Montel']['predict_path'])
h_cols = [c for c in montel_pred.columns if c.startswith("h")]
montel_window_match = len(h_cols) == config['data']['prediction_window']

# 2. 读取自己模型的交叉验证结果
model_name = config['model_name']
cv_results_path = f'outputs/{model_name}/cv_results_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'
cv_results = pd.read_csv(cv_results_path)

if montel_window_match:
    # ---- 窗口一致：Montel 和模型一起对比 ----

    # 3. 找出共同的 begin_utc
    common_times = set(montel_pred['begin_utc']) & set(cv_results['begin_utc'])
    print(f"\n共同时间点数量: {len(common_times)}")

    # 4. 各自只保留共同时间点的数据
    montel_filtered = montel_pred[montel_pred['begin_utc'].isin(common_times)].sort_values('begin_utc').reset_index(drop=True)
    model_filtered = cv_results[cv_results['begin_utc'].isin(common_times)].sort_values('begin_utc').reset_index(drop=True)
    print(montel_filtered.head())
    print(model_filtered.head())
    h_cols = [c for c in montel_filtered.columns if c.startswith("h")]
    h_df = montel_filtered[h_cols]

    montel_predict = h_df.values.flatten()
    ground_truth = model_filtered['y'].values.flatten()
    model_predict = model_filtered[model_name].values.flatten()
    time_array = model_filtered['ds'].values

    montel_ground = {
        'mae': mae(ground_truth, montel_predict),
        'rmse': rmse(ground_truth, montel_predict),
        'bias': bias(ground_truth, montel_predict),
        'wape': wape(ground_truth, montel_predict),
    }

    model_ground = {
        'mae': mae(ground_truth, model_predict),
        'rmse': rmse(ground_truth, model_predict),
        'bias': bias(ground_truth, model_predict),
        'wape': wape(ground_truth, model_predict),
    }

    df_metrics = pd.DataFrame(
        [montel_ground, model_ground],
        index=['Montel', model_name]
    )
else:
    # ---- 窗口不一致：只处理模型数据 ----
    print(f"\n警告: Montel 窗口数量 ({len(h_cols)}) 与 prediction_window ({config['data']['prediction_window']}) 不一致，仅处理模型数据")

    ground_truth = cv_results['y'].values.flatten()
    model_predict = cv_results[model_name].values.flatten()
    time_array = cv_results['ds'].values
    montel_predict = None

    model_ground = {
        'mae': mae(ground_truth, model_predict),
        'rmse': rmse(ground_truth, model_predict),
        'bias': bias(ground_truth, model_predict),
        'wape': wape(ground_truth, model_predict),
    }

    df_metrics = pd.DataFrame([model_ground], index=[model_name])

# 保存为 CSV
df_metrics.to_csv(f'outputs/{model_name}/compare_metrics.csv', index=True)

# 再随机读取几个窗口进行可视化
plot_windows_with_time(
    ground_truth, montel_predict, model_predict, time_array,
    horizon=config['data']['prediction_window'], k=config['Montel']['compare_plot_k'],
    save_dir=f'outputs/{model_name}/window_plots', show=True, rename=model_name
)