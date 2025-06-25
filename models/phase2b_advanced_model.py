# -*- coding: utf-8 -*-
"""
Phase 2B: Advanced Model Training Pipeline
"""
import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as make_imblearn_pipeline

def load_featured_data(file_path):
    """
    載入包含特徵和標籤的 Parquet 資料。
    """
    if not os.path.exists(file_path):
        print(f"錯誤：找不到資料檔案 -> {file_path}")
        return None
    print(f"從 {file_path} 載入資料...")
    df = pd.read_parquet(file_path)
    df.dropna(subset=['label'], inplace=True) # 確保標籤存在
    print("資料載入完成。")
    return df

def create_advanced_features(df):
    """
    根據 PROJECT_PLAN.md 2.B.1 建立進階時序特徵。
    """
    print("正在建立進階特徵...")
    df_feat = df.copy()
    
    # 1. 均線斜率
    for k in [5, 10, 20]:
        df_feat[f'SMA_20_slope_{k}'] = (df_feat['SMA_20'] - df_feat['SMA_20'].shift(k)) / k
        df_feat[f'EMA_20_slope_{k}'] = (df_feat['EMA_20'] - df_feat['EMA_20'].shift(k)) / k

    # 2. VWAP 偏離 (假設已有 VWAP 欄位，若無則跳過)
    if 'VWAP' in df_feat.columns:
        df_feat['vwap_dev'] = (df_feat['close'] - df_feat['VWAP']) / df_feat['VWAP']

    # 3. 前 N tick 極值
    for n in [5, 10, 20]:
        df_feat[f'high_last_{n}'] = df_feat['close'].rolling(window=n).max()
        df_feat[f'low_last_{n}'] = df_feat['close'].rolling(window=n).min()
        df_feat[f'range_last_{n}'] = df_feat[f'high_last_{n}'] - df_feat[f'low_last_{n}']

    # 4. 成交量加權指標
    if 'volume' in df_feat.columns:
        df_feat['vol_ratio_20'] = df_feat['volume'] / df_feat['volume'].rolling(window=20).mean()

    # 處理因計算產生的 NaN 值
    df_feat.bfill(inplace=True)
    df_feat.fillna(0, inplace=True)
    print("進階特徵建立完成。")
    return df_feat

def get_timeseries_split(df, n_splits=5, test_size=None):
    """
    使用 TimeSeriesSplit 取得最後一折的訓練/測試集索引。
    """
    print("正在進行時序資料切分...")
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    all_splits = list(tscv.split(df))
    train_idx, test_idx = all_splits[-1]
    print(f"訓練集大小: {len(train_idx)}")
    print(f"測試集大小: {len(test_idx)}")
    return train_idx, test_idx

def train_advanced_model(X_train, y_train):
    """
    訓練一個 XGBoost 模型，並使用 imblearn pipeline 處理不平衡問題。
    """
    print("正在訓練 XGBoost 模型...")
    
    # 定義 XGBoost 分類器
    xgb_classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # 使用 imblearn 的 pipeline 來整合過採樣和分類器
    # 注意：過採樣只應作用於訓練資料
    model_pipeline = make_imblearn_pipeline(
        RandomOverSampler(random_state=42),
        xgb_classifier
    )
    
    model_pipeline.fit(X_train, y_train)
    print("模型訓練完成。")
    return model_pipeline

def evaluate_model(model, X_test, y_test, features):
    """
    評估模型並印出分類報告。
    """
    print("\n--- 模型評估報告 ---")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Down (-1)', 'Neutral (0)', 'Up (1)'])
    print(report)
    print("--------------------\n")
    
    # 顯示特徵重要性
    try:
        feature_importances = model.steps[-1][1].feature_importances_
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        print("--- Top 10 特徵重要性 ---")
        print(importance_df.head(10))
        print("-------------------------\n")
    except Exception as e:
        print(f"無法取得特徵重要性: {e}")


def save_model(model, save_path):
    """
    儲存訓練好的模型。
    """
    model_dir = os.path.dirname(save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    joblib.dump(model, save_path)
    print(f"模型已儲存至: {save_path}")

def run_advanced_training_pipeline(data_path, model_save_path):
    """
    執行完整的進階模型訓練流程。
    """
    # 1. 載入資料
    df = load_featured_data(data_path)
    if df is None:
        return

    # 2. 建立進階特徵
    df_featured = create_advanced_features(df)
    
    # 3. 定義特徵與目標
    base_features = ['SMA_20', 'EMA_20', 'RSI_14', 'BB_Middle', 'BB_Upper', 'BB_Lower']
    advanced_features = [col for col in df_featured.columns if 'slope' in col or 'dev' in col or 'last' in col or 'ratio' in col]
    features = base_features + advanced_features
    target = 'label'

    # 確保所有特徵都存在
    features = [f for f in features if f in df_featured.columns]
    print(f"使用的特徵數量: {len(features)}")

    X = df_featured[features]
    y = df_featured[target]

    # 將標籤從 [-1, 0, 1] 映射到 [0, 1, 2] 以符合 XGBoost 的要求
    y = y.map({-1: 0, 0: 1, 1: 2})

    # 4. 時序資料切分
    train_idx, test_idx = get_timeseries_split(df_featured, n_splits=5, test_size=int(len(df_featured) * 0.15))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 5. 訓練模型
    model = train_advanced_model(X_train, y_train)

    # 6. 評估模型
    evaluate_model(model, X_test, y_test, features)

    # 7. 儲存模型
    save_model(model, model_save_path)

if __name__ == '__main__':
    # 當作獨立腳本執行時的進入點
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 設定資料路徑和模型儲存路徑
    DATA_PATH = os.path.join(project_root, 'data', 'processed', 'ticks_2025-0527_with_features.parquet')
    MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'trained_models', 'advanced_xgboost_model.joblib')

    run_advanced_training_pipeline(data_path=DATA_PATH, model_save_path=MODEL_SAVE_PATH)