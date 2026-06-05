import pandas as pd
import numpy as np
import lightgbm as lgb
from .base import BaseModel
from .model_registry import register_model
from sklearn.metrics import mean_absolute_error

@register_model("LightGBM")
class LightGBMModel(BaseModel):
    """LightGBM 回归模型，只使用时间特征预测电价。"""

    def __init__(self, config, model_config):
        self.config = config
        self.model_name = config['model_name']
        self.target_col = config['data']['target_col']
        self.time_col = config['data']['raw_col'][0]
        self.freq = config['data']['freq']

        # 特征列：只取配置里指定的时间特征
        time_features = config['data']['feature_kwargs'].get('time_features', []) or []
        self.feature_cols = list(time_features)

        # LightGBM 参数
        model_params = dict(model_config['LightGBM'].get('params', {}))
        self.lgb_params = model_params

        self.model = lgb.LGBMRegressor(**self.lgb_params)

        # 时间划分
        self.train_start = pd.to_datetime(config['data']['train']['start']).tz_localize('UTC')
        self.val_start = pd.to_datetime(config['data']['val']['start']).tz_localize('UTC')
        self.test_start = pd.to_datetime(config['data']['test']['start']).tz_localize('UTC')
        self.test_end = pd.to_datetime(config['data']['test']['end']).tz_localize('UTC')

    # ─────────── 训练 ───────────

    def fit(self, df_full: pd.DataFrame):
        #输出df的前五行
        # print(df.head())
        # pass
        df = df_full.sort_values(self.time_col).reset_index(drop=True)

        # 按时间划分
        train = df[df[self.time_col] < self.val_start]
        val = df[(df[self.time_col] >= self.val_start) & (df[self.time_col] < self.test_start)]

        X_train, y_train = train[self.feature_cols], train[self.target_col]
        X_val, y_val = val[self.feature_cols], val[self.target_col]

        print(f"特征列 ({len(self.feature_cols)}): {self.feature_cols}")
        print(f"训练集: {len(X_train)} 条")
        print(f"验证集: {len(X_val)} 条")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        # # 打印验证集结果
        y_pred = self.model.predict(X_val)
        print(f"验证集 MAE: {mean_absolute_error(y_val, y_pred):.4f}")
        return self

    # ─────────── 预测 ───────────

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        #打印future_df的全部数据
        print(future_df)
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        X = future_df[self.feature_cols]
        yhat = self.model.predict(X)
        return pd.DataFrame({"ds": future_df[self.time_col], self.model_name: yhat})

    # ─────────── 交叉验证 ───────────

    def cross_validate(self, df_full: pd.DataFrame):
        """滑动窗口交叉验证。

        模型只训练一次（不存在则自动调用 fit()），
        然后在测试集上以 prediction_window 为窗口大小、步长 1h 滑动预测。
        """
        # 如果还没训练，先训练
        if self.model is None or not hasattr(self.model, 'booster_') or self.model.booster_ is None:
            print("模型未训练，自动调用 fit()...")
            self.fit(df_full)

        df = df_full.sort_values(self.time_col).reset_index(drop=True)

        prediction_window = self.config['data']['prediction_window']
        horizon_total = self.config.get('horizon_total', prediction_window)  # test.py 设置为了 1080
        local_tz = self.config['data']['feature_kwargs']['local_tz']

        # 测试集（严格按配置的时间范围）
        test_df = df[(df[self.time_col] >= self.test_start) & (df[self.time_col] <= self.test_end)].copy()
        test_df = test_df.reset_index(drop=True)

        # 找到每天 0 点作为窗口起点
        test_df['ds_local'] = test_df[self.time_col].dt.tz_convert(local_tz)
        midnight_mask = test_df['ds_local'].dt.hour == 0
        start_indices = test_df.index[midnight_mask].tolist()
        
        total = len(start_indices)
        print(f"滑动窗口预测: {total} 个窗口（每天 0 点开始，窗口大小 {prediction_window}h，输入 {horizon_total}h）")

        all_rows = []
        for idx, start_idx in enumerate(start_indices):
            end_idx = start_idx + horizon_total  # 取 1080 条输入
            if end_idx > len(test_df):
                break

            window = test_df.iloc[start_idx:end_idx]
            y_pred = self.model.predict(window[self.feature_cols])
            # 只保留后 prediction_window 条（1056）
            y_pred = y_pred[-prediction_window:]
            window = window.tail(prediction_window)
            window_start = window[self.time_col].iloc[0]  # 窗口起点（当天 0 点）

            for j, (_, row) in enumerate(window.iterrows()):
                all_rows.append({
                    "ds": row[self.time_col],
                    "y": row[self.target_col],
                    self.model_name: y_pred[j],
                    "cutoff": window_start,
                })

            if (idx + 1) % max(1, total // 10) == 0 or idx == total - 1:
                print(f"  进度: {idx + 1}/{total}")

        cv_results = pd.DataFrame(all_rows)
        if cv_results.empty:
            print("警告: 没有产生任何窗口")
            return cv_results

        cv_results['ds'] = pd.to_datetime(cv_results['ds'])
        cv_results['cutoff'] = pd.to_datetime(cv_results['cutoff'])
        cv_selected = cv_results.copy()
        cv_selected["begin_utc"] = cv_selected.groupby("cutoff")["ds"].transform("first")
        cv_selected = cv_selected.drop(columns=["cutoff"])

        return cv_selected.reset_index(drop=True)


    # ─────────── 保存 / 加载 ───────────

    def save(self, path: str):
        import os, joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path: str, config=None, model_config=None):
        """加载模型。

        Args:
            path: 模型文件路径
            config: 配置字典（必需，因为 __init__ 依赖它初始化参数）
        """
        if config is None:
            raise ValueError("加载模型时必须提供 config 参数")
        if model_config is None:
            raise ValueError("加载模型时必须提供 model_config 参数")
        import joblib
        obj = cls(config, model_config)
        data = joblib.load(path)
        obj.model = data["model"]
        obj.feature_cols = data["feature_cols"]
        return obj
