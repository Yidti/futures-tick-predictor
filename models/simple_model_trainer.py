import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

from utils.data_splitter import split_data
from imblearn.over_sampling import RandomOverSampler  # pip install imbalanced-learn

def train_and_save_simple_model(df: pd.DataFrame, features: list, target: str, model_path: str = 'trained_models/simple_model.pkl'):
    """
    訓練一個以 RandomForest 為基底的三元分類模型，並儲存。
    同時示範如何針對少數類做過採樣 (oversampling)。

    Args:
        df (pd.DataFrame): 包含特徵和目標的 DataFrame。
        features (list): 特徵欄位名稱列表。
        target (str): 目標欄位名稱（-1, 0, 1）。
        model_path (str): 模型儲存路徑。
    """
    print(f"開始訓練三元分類模型（RandomForest），特徵: {features}, 目標: {target}")

    # ----------------------------------------------------------------------
    # 1. 先拆訓練/測試集
    X_train, X_test, y_train, y_test = split_data(df, features, target)

    # ----------------------------------------------------------------------
    # 2. 對訓練集做過採樣 (只處理 X_train, y_train)
    #    這裡用 RandomOverSampler，把 -1 與 1 兩個類別都重複取樣成跟 0 類一樣多
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    print("◎ 過採樣後的 label 分佈：")
    print(pd.Series(y_resampled).value_counts())

    # ----------------------------------------------------------------------
    # 3. 建 pipeline（先 imputer 再 RandomForest）
    imputer = SimpleImputer(strategy='mean')
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced_subsample'  # 再次平衡，雖然已經做過採樣，但保留防止偏差
    )
    model = Pipeline([
        ('imputer', imputer),
        ('classifier', rf)
    ])

    # ----------------------------------------------------------------------
    # 4. 訓練模型
    model.fit(X_resampled, y_resampled)

    # ----------------------------------------------------------------------
    # 5. 在測試集上評估
    y_pred = model.predict(X_test)
    print("=== 測試集上的分類報告 (classification_report) ===")
    print(classification_report(y_test, y_pred, digits=4))

    # ----------------------------------------------------------------------
    # 6. 儲存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"模型已成功儲存至: {model_path}")

if __name__ == '__main__':
    # 測試用範例資料 (實際請用真實 tick 時序資料)
    data = {
        'ts': pd.to_datetime([
            '2023-01-01 09:00:00', '2023-01-01 09:00:01', '2023-01-01 09:00:02',
            '2023-01-01 09:00:03', '2023-01-01 09:00:04', '2023-01-01 09:00:05',
            '2023-01-01 09:00:06', '2023-01-01 09:00:07', '2023-01-01 09:00:08',
            '2023-01-01 09:00:09', '2023-01-01 09:00:10', '2023-01-01 09:00:11',
            '2023-01-01 09:00:12', '2023-01-01 09:00:13', '2023-01-01 09:00:14'
        ]),
        'close':    [100, 101, 102, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        'volume':   [1000,1200,1100,1300,1050,1150,1250,1350,1450,1550,1650,1750,1850,1950,2050],
        'bid_price':[99,100,101,100,101,102,103,104,105,106,107,108,109,110,111],
        'bid_volume':[900,1100,1000,1200,950,1050,1150,1250,1350,1450,1550,1650,1750,1850,1950],
        'ask_price':[101,102,103,102,103,104,105,106,107,108,109,110,111,112,113],
        'ask_volume':[1100,1300,1200,1400,1150,1250,1350,1450,1550,1650,1750,1850,1950,2050,2150],
        'tick_type':[1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
        'is_traffic_limited':[0]*15,
        'SMA_20':[100,100.5,101,101.2,101.5,101.8,102.1,102.4,102.7,103,103.3,103.6,103.9,104.2,104.5],
        'EMA_20':[100,100.6,101.1,101.3,101.6,101.9,102.2,102.5,102.8,103.1,103.4,103.7,104,104.3,104.6],
        'RSI_14':[50,55,60,58,62,65,68,70,72,75,78,80,82,85,88],
        'BB_Middle':[100,100.5,101,101.2,101.5,101.8,102.1,102.4,102.7,103,103.3,103.6,103.9,104.2,104.5],
        'BB_Upper':[101,101.5,102,102.2,102.5,102.8,103.1,103.4,103.7,104,104.3,104.6,104.9,105.2,105.5],
        'BB_Lower':[99,99.5,100,100.2,100.5,100.8,101.1,101.4,101.7,102,102.3,102.6,102.9,103.2,103.5],
        'label': [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
    }
    df = pd.DataFrame(data)

    features_to_use = [
        'close','volume','bid_price','bid_volume','ask_price','ask_volume',
        'tick_type','is_traffic_limited','SMA_20','EMA_20','RSI_14',
        'BB_Middle','BB_Upper','BB_Lower'
    ]
    target_label = 'label'

    train_and_save_simple_model(df, features_to_use, target_label)
