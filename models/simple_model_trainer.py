import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

# 假設 utils 模組在 PYTHONPATH 中，或者在同一個專案結構下
from utils.data_splitter import split_data

def train_and_save_simple_model(df: pd.DataFrame, features: list, target: str, model_path: str = 'models/simple_model.pkl'):
    """
    訓練一個簡單的 Logistic Regression 模型並儲存。

    Args:
        df (pd.DataFrame): 包含特徵和目標的 DataFrame。
        features (list): 特徵欄位名稱列表。
        target (str): 目標欄位名稱。
        model_path (str): 模型儲存路徑。
    """
    print(f"開始訓練簡易模型，特徵: {features}, 目標: {target}")

    # 資料切分
    X_train, X_test, y_train, y_test = split_data(df, features, target)

    # 建立一個包含 Imputer 和 Logistic Regression 的 Pipeline
    # 使用 mean 策略填充缺失值
    imputer = SimpleImputer(strategy='mean')
    model = Pipeline([
        ('imputer', imputer),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, class_weight='balanced')) # 增加 max_iter 以避免收斂警告，並加入 class_weight='balanced' 處理類別不平衡
    ])
    model.fit(X_train, y_train)

    # 評估模型 (可選)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型在測試集上的準確度: {accuracy:.4f}")

    # 確保模型儲存目錄存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 儲存模型
    joblib.dump(model, model_path)
    print(f"模型已成功儲存至: {model_path}")

if __name__ == '__main__':
    # 範例資料 (實際應用中會從資料載入模組獲取)
    data = {
        'ts': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-01 09:00:01', '2023-01-01 09:00:02', '2023-01-01 09:00:03', '2023-01-01 09:00:04', '2023-01-01 09:00:05', '2023-01-01 09:00:06', '2023-01-01 09:00:07', '2023-01-01 09:00:08', '2023-01-01 09:00:09', '2023-01-01 09:00:10', '2023-01-01 09:00:11', '2023-01-01 09:00:12', '2023-01-01 09:00:13', '2023-01-01 09:00:14']),
        'close': [100, 101, 102, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        'volume': [1000, 1200, 1100, 1300, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050],
        'bid_price': [99, 100, 101, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'bid_volume': [900, 1100, 1000, 1200, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950],
        'ask_price': [101, 102, 103, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        'ask_volume': [1100, 1300, 1200, 1400, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150],
        'tick_type': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        'is_traffic_limited': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SMA_20': [100, 100.5, 101, 101.2, 101.5, 101.8, 102.1, 102.4, 102.7, 103, 103.3, 103.6, 103.9, 104.2, 104.5],
        'EMA_20': [100, 100.6, 101.1, 101.3, 101.6, 101.9, 102.2, 102.5, 102.8, 103.1, 103.4, 103.7, 104, 104.3, 104.6],
        'RSI_14': [50, 55, 60, 58, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 88],
        'BB_Middle': [100, 100.5, 101, 101.2, 101.5, 101.8, 102.1, 102.4, 102.7, 103, 103.3, 103.6, 103.9, 104.2, 104.5],
        'BB_Upper': [101, 101.5, 102, 102.2, 102.5, 102.8, 103.1, 103.4, 103.7, 104, 104.3, 104.6, 104.9, 105.2, 105.5],
        'BB_Lower': [99, 99.5, 100, 100.2, 100.5, 100.8, 101.1, 101.4, 101.7, 102, 102.3, 102.6, 102.9, 103.2, 103.5],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    features_to_use = ['close', 'volume', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'tick_type', 'is_traffic_limited', 'SMA_20', 'EMA_20', 'RSI_14', 'BB_Middle', 'BB_Upper', 'BB_Lower']
    target_label = 'label'

    train_and_save_simple_model(df, features_to_use, target_label)