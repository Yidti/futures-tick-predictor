# -*- coding: utf-8 -*-
"""
Phase 2A: Simple Model Training Pipeline
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load_featured_data(file_path):
    """
    載入包含特徵和標籤的 Parquet 資料。

    Args:
        file_path (str): 資料檔案的路徑。

    Returns:
        pd.DataFrame: 載入的 DataFrame，如果檔案不存在則返回 None。
    """
    if not os.path.exists(file_path):
        print(f"錯誤：找不到資料檔案 -> {file_path}")
        return None
    print(f"從 {file_path} 載入資料...")
    df = pd.read_parquet(file_path)
    df.dropna(inplace=True) # 確保沒有 NaN 值影響模型訓練
    print("資料載入完成。")
    return df

def split_data(df, features, target, test_size=0.2, random_state=42):
    """
    將資料切分為訓練集和測試集。

    Args:
        df (pd.DataFrame): 包含特徵和目標的 DataFrame。
        features (list): 要使用的特徵欄位名稱列表。
        target (str): 目標欄位名稱。
        test_size (float): 測試集所佔的比例。
        random_state (int): 隨機種子。

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("正在切分資料...")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"訓練集大小: {X_train.shape[0]}")
    print(f"測試集大小: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def train_simple_model(X_train, y_train):
    """
    訓練一個簡單的邏輯回歸模型。

    Args:
        X_train (pd.DataFrame): 訓練特徵。
        y_train (pd.Series): 訓練目標。

    Returns:
        sklearn.pipeline.Pipeline: 訓練好的模型 Pipeline。
    """
    print("正在訓練簡易模型...")
    simple_model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    simple_model_pipeline.fit(X_train, y_train)
    print("模型訓練完成。")
    return simple_model_pipeline

def evaluate_model(model, X_test, y_test):
    """
    評估模型並印出分類報告。

    Args:
        model (sklearn.pipeline.Pipeline): 訓練好的模型。
        X_test (pd.DataFrame): 測試特徵。
        y_test (pd.Series): 測試目標。
    """
    print("\n--- 模型評估報告 ---")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Down (-1)', 'Neutral (0)', 'Up (1)'])
    print(report)
    print("--------------------\n")

def save_model(model, save_path):
    """
    儲存訓練好的模型。

    Args:
        model (sklearn.pipeline.Pipeline): 要儲存的模型。
        save_path (str): 儲存路徑。
    """
    model_dir = os.path.dirname(save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    joblib.dump(model, save_path)
    print(f"模型已儲存至: {save_path}")

def run_training_pipeline(data_path, model_save_path):
    """
    執行完整的簡易模型訓練流程。

    Args:
        data_path (str): 原始資料路徑。
        model_save_path (str): 模型儲存路徑。
    """
    # 1. 載入資料
    df = load_featured_data(data_path)
    if df is None:
        return

    # 2. 選擇特徵與目標
    simple_features = ['SMA_20', 'EMA_20', 'RSI_14', 'BB_Middle', 'BB_Upper', 'BB_Lower']
    target = 'label'
    
    # 檢查所需欄位是否存在
    missing_cols = [col for col in simple_features + [target] if col not in df.columns]
    if missing_cols:
        print(f"錯誤：資料中缺少以下必要欄位: {missing_cols}")
        return

    # 3. 資料切分
    X_train, X_test, y_train, y_test = split_data(df, simple_features, target)

    # 4. 訓練模型
    model = train_simple_model(X_train, y_train)

    # 5. 評估模型
    evaluate_model(model, X_test, y_test)

    # 6. 儲存模型
    save_model(model, model_save_path)

if __name__ == '__main__':
    # 當作獨立腳本執行時的進入點
    # 設定相對路徑
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 設定資料路徑和模型儲存路徑
    DATA_PATH = os.path.join(project_root, 'data', 'processed', 'ticks_2025-04_with_features.parquet')
    MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'trained_models', 'simple_logistic_model.joblib')

    run_training_pipeline(data_path=DATA_PATH, model_save_path=MODEL_SAVE_PATH)