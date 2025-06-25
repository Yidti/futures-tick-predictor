# 🚀 金融期貨 Tick 資料即時預測系統 – Side Project 詳細執行計畫 🚀

**專案總目標：**
建立一個能針對台指期等 Tick / 分K 資料進行 **即時特徵工程 + 預測 + 回測驗證** 的 **可視化預測系統**。
最終部署為 Streamlit 應用，未來可銜接 WebSocket 即時資料與真實交易 API。

---

## 目錄

- [🚀 金融期貨 Tick 資料即時預測系統 – Side Project 詳細執行計畫 🚀](#-金融期貨-tick-資料即時預測系統--side-project-詳細執行計畫-)
  - [目錄](#目錄)
  - [核心技術棧](#核心技術棧)
  - [⓪ 專案初始化與環境設定](#-專案初始化與環境設定)
  - [① 第一階段：資料準備與探索 (Data Preparation \& EDA)](#-第一階段資料準備與探索-data-preparation--eda)
  - [②.A 快速原型：簡易模型與前端展示 (Rapid Prototype: Simple Model \& Frontend)](#a-快速原型簡易模型與前端展示-rapid-prototype-simple-model--frontend)
  - [②.B 第二階段：模型訓練與預測模組 (進階) (Model Training \& Prediction – Advanced)](#b-第二階段模型訓練與預測模組-進階-model-training--prediction--advanced)
    - [2.B.1 模型選擇與特徵設計](#2b1-模型選擇與特徵設計)
      - [1. 為何單純 RandomForest（或一般 tree-based）不夠？](#1-為何單純-randomforest或一般-tree-based不夠)
      - [2. 改進方案：適合時序的模型選擇](#2-改進方案適合時序的模型選擇)
      - [3. 加入「跨時間點」的核心特徵](#3-加入跨時間點的核心特徵)
    - [2.B.2 標註欄位設計 (Labeling)](#2b2-標註欄位設計-labeling)
      - [1. 常見標註方式](#1-常見標註方式)
      - [2. 範例：未來 5 tick 方向分類](#2-範例未來-5-tick-方向分類)
      - [3. 動態上下界](#3-動態上下界)
      - [4. 多階段標註](#4-多階段標註)
    - [2.B.3 時序資料切分 (Time-Series Split)](#2b3-時序資料切分-time-series-split)
      - [更進階切分策略](#更進階切分策略)
    - [2.B.4 訓練與驗證 Pipeline 範例](#2b4-訓練與驗證-pipeline-範例)
  - [③ 第三階段：回測與績效驗證 (Backtesting \& Performance Validation)](#-第三階段回測與績效驗證-backtesting--performance-validation)
    - [3.1 回測邏輯示例](#31-回測邏輯示例)
  - [④ 第四階段：前端功能完善與即時化 (Frontend Enhancement \& Real-time Integration)](#-第四階段前端功能完善與即時化-frontend-enhancement--real-time-integration)
  - [⑤ 第五階段：自動化與部署 (Automation \& Deployment) *(選做/遠期)*](#-第五階段自動化與部署-automation--deployment-選做遠期)
  - [📊 模組互動示意圖 (Mermaid)](#-模組互動示意圖-mermaid)
  - [參考資料與工具推薦](#參考資料與工具推薦)

---

## 核心技術棧

* **資料處理與分析：** `pandas`, `numpy`
* **技術指標計算：** `ta-lib`
* **機器學習/深度學習：**

  * 初期：`scikit-learn`, `xgboost`
  * 進階：`tensorflow/keras`, `pytorch` (LSTM/Transformer)
* **模型解釋性：** `shap`
* **回測框架：** 簡易自建，或考慮 `backtrader`, `bt`
* **前端與可視化：** `streamlit`, `plotly`, `matplotlib`
* **版本控制：** `Git`
* **開發環境：** `Jupyter Notebook` (EDA 與實驗), Python IDE

---

## ⓪ 專案初始化與環境設定

| 任務編號 | 任務內容                       | 預期產出                                                                                                                        | 涉及工具/模組                   | 備註           |
| ---- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------ |
| 0.1  | 建立專案根目錄及子目錄結構              | 符合專案目錄結構 (`data/`, `features/`, `models/`, `backtest/`, `app/`, `notebooks/`, `utils/`)                                     | 檔案系統                      |              |
| 0.2  | 初始化 Git 版本控制               | `.git` 目錄，專案納入版本控制                                                                                                          | `git`                     |              |
| 0.3  | 建立 `requirements.txt`      | 列出初步依賴套件（`pandas`, `numpy`, `scikit-learn`, `xgboost`, `ta-lib`, `streamlit`, `jupyterlab`, `matplotlib`, `plotly`, `shap`） | 文字編輯器                     |              |
| 0.4  | 設定 Python 虛擬環境             | 啟用 venv 或 conda 環境                                                                                                          | `venv` / `conda`          | 強烈建議隔離專案依賴   |
| 0.5  | 安裝初步依賴套件                   | `pip install -r requirements.txt`                                                                                           | `pip`                     |              |
| 0.6  | 設定 Jupyter Notebook/Lab 環境 | 能正常啟動並運行 Jupyter Lab/Notebook                                                                                               | `jupyterlab` / `notebook` | 用於後續 EDA 與實驗 |

---

## ① 第一階段：資料準備與探索 (Data Preparation & EDA)

| 任務編號 | 任務內容                         | 預期產出                                                                                           | 涉及工具/模組                                               | 備註                               |
| ---- | ---------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------- |
| 1.1  | **資料載入與初步檢視**                | Python 腳本（`.py` 或 `.ipynb`）可成功讀取 CSV/Parquet Tick 歷史資料                                         | `pandas`, `notebooks/`                                | 確認欄位名稱、資料型態、筆數、時間範圍              |
| 1.2  | **資料清洗**                     | 清洗後 DataFrame：處理缺失值 (NaN)、異常價格 (outliers)、重複資料                                                 | `pandas`, `numpy`, `notebooks/`                       | 記錄清洗策略與過程                        |
| 1.3  | **時間序列處理**                   | DataFrame 設定時間欄位為 index；時間格式標準化；過濾非交易時間                                                        | `pandas`, `notebooks/`                                | 如盤後交易需分開處理                       |
| 1.4  | **初步探索性資料分析 (EDA)**          | `notebooks/01_eda.ipynb` 包含：<br> - 價格走勢 (開高低收) <br> - 成交量變化 <br> - 波動率分析 (日報酬標準差) <br> - 價量分佈圖 | `pandas`, `matplotlib`, `seaborn`, `plotly` (可選)      |                                  |
| 1.5  | **技術指標特徵工程 (基礎)**            | DataFrame 新增核心技術指標欄位 (如 SMA、EMA、RSI)                                                           | `ta-lib`, `pandas`, `features/feature_engineering.py` | 選擇少量易懂指標快速原型                     |
| 1.6  | **標註欄位製作 (Target Labeling)** | DataFrame 新增目標預測欄位 (如 `target_price_up_N_ticks`)，值為 0/1                                        | `pandas`, `numpy`, `features/labeling.py`             | 明確定義預測目標 (如「未來 N tick 後漲跌幅 > Δ」) |

---

## ②.A 快速原型：簡易模型與前端展示 (Rapid Prototype: Simple Model & Frontend)

| 任務編號  | 任務內容                | 預期產出                                                                    | 涉及工具/模組                                             | 備註            |
| ----- | ------------------- | ----------------------------------------------------------------------- | --------------------------------------------------- | ------------- |
| 2.A.1 | **資料切分 (簡易)**       | 撰寫函數將資料隨機切分為訓練集與測試集，用於原型                                                | `scikit-learn`                                      |               |
| 2.A.2 | **訓練極簡模型**          | 訓練一個非常簡單的分類模型（如 Logistic Regression、Dummy Classifier 或 XGBoost），並儲存模型檔案 | `scikit-learn` / `xgboost`, `joblib` / `pickle`     | 目標跑通流程，非追求高精度 |
| 2.A.3 | **Streamlit UI 雛形** | 建立 `app/main_app.py`，包含：<br> - 頁面標題與基本 Layout <br> - CSV 資料上傳 (或內建範例資料) | `streamlit`                                         |               |
| 2.A.4 | **簡易模型推論整合**        | Streamlit 應用能載入已訓練的極簡模型，對上傳或內建資料進行預測                                    | `streamlit`, `joblib` / `pickle`                    |               |
| 2.A.5 | **前端結果初步展示**        | Streamlit 頁面可顯示：<br> - 輸入資料摘要或基礎圖表 <br> - 模型預測結果 (如預測上漲/下跌)             | `streamlit`, `pandas`, `matplotlib` / `plotly` (可選) | 讓使用者看到端到端互動   |

---

## ②.B 第二階段：模型訓練與預測模組 (進階) (Model Training & Prediction – Advanced)

### 2.B.1 模型選擇與特徵設計

#### 1. 為何單純 RandomForest（或一般 tree-based）不夠？

1. **假設樣本獨立 (iid)，無法捕捉長期時序依賴**

   * Tick 資料具強烈時間序列特性，未考慮順序關係的模型容易忽略市場動能或反轉訊號。
2. **缺少「跨時間點」的核心特徵**

   * 若僅餵「當下 snapshot」(如 close, volume) 給 RF，模型只能學到當前資訊的瞬時關係，卻難捕捉過去 N 筆 tick 的連續變化趨勢。
3. **Label 設計過於粗糙／單純**

   * 單純以 +1 / 0 / -1 類別標註漲跌，可能忽略「價差幅度是否足以交易」的判斷，以及不同盤勢下行為模式的差異。

#### 2. 改進方案：適合時序的模型選擇

* **LSTM / GRU / Transformer 時序神經網路**

  * LSTM：擅長捕捉長短期記憶，適合序列預測
  * Temporal Fusion Transformer (TFT)：可同時處理多種時序特徵，動態學習注意力權重
  * TCN (Temporal Convolutional Network)：捲積方式學習時序依賴，相對易訓練且速度較快

* **TSFresh + Tree-based**

  * 用 TSFresh 自動萃取上百種時序統計特徵（如平均值、變異數、非線性指標），再餵給 XGBoost/LightGBM
  * 比單純 snapshot 更能量化時序模式、異常點等

* **多階段架構**

  1. **階段一：盤勢分類器**

     * 輸入特徵：當前價格、短期技術指標（SMA 斜率、RSI、KDJ）、成交量突增…
     * 輸出：`{震盪盤, 低波動, 高波動, 上漲趨勢, 下跌趨勢}`
     * 模型可用 LightGBM、XGBoost、簡單神經網路
     * 目的：先釐清「盤勢背景」，讓後續 Tick 級預測有條件限制
  2. **階段二：Tick 頻預測**

     * 若階段一判定為「高波動趨勢」，使用 LSTM/Transformer 預測未來 N tick 價差
     * 若為「震盪盤」，則先判斷「是否突破震盪區間」再決策進出
     * 輸出：`{進場多, 進場空, 觀望}`，並附帶「預期持有時長」、「停損停利建議」

#### 3. 加入「跨時間點」的核心特徵

1. **均線斜率（Moving Average Slope）**

   * `SMA_20_slope = (SMA_20 - SMA_20.shift(k)) / k` (k 可為 5、10、20 tick)
   * `EMA_10_slope = (EMA_10 - EMA_10.shift(k)) / k`
2. **VWAP 偏離 (VWAP Deviation)**

   * `vwap_dev = (price - VWAP_1min) / VWAP_1min`
   * 若 VWAP 取自 1 分鐘或 5 分鐘 K 線，能反映機構買賣力道
3. **前 N 分鐘極值**

   * `high_last5 = price.rolling(window=5).max()`
   * `low_last5 = price.rolling(window=5).min()`
   * 差值：`range_last5 = high_last5 - low_last5`
4. **成交量加權指標**

   * `vol_ratio = volume / volume.rolling(window=20).mean()`
   * 若瞬間量能突增 (vol\_ratio > 2)，可能為機構進出信號
5. **時間區段標籤**

   * `is_opening = timestamp.between('09:00:00','09:05:00')`
   * `is_midday = timestamp.between('11:30:00','13:25:00')`
   * `is_closing = timestamp.between('13:25:00','13:30:00')`

---

### 2.B.2 標註欄位設計 (Labeling)

> \*\*目標：\*\*根據策略需求，為每個 Tick/分K 資料點產生可學習且可行的預測目標。

#### 1. 常見標註方式

1. **未來價格漲跌分類**

   * 例如：`label = 1` (未來 N tick 後收盤 > 當前價 + Δ)，`label = -1` (未來 N tick 後收盤 < 當前價 − Δ)，`label = 0` (無顯著變動)。
   * Δ 可設定為固定點數 (如 1 點) 或基於波動率動態計算。
2. **未來報酬率分類**

   * `label = 1` if `(close.shift(-N) - close) / close > X%`，否則 `label = 0`。
3. **回歸型標註**

   * 直接預測未來 N tick 後的價格或報酬值 (連續值)。

#### 2. 範例：未來 5 tick 方向分類

```python
def generate_label_tick(df: pd.DataFrame, n_tick: int = 5, delta: float = 0.0):
    """
    為每筆 Tick 資料標註未來 n_tick 收盤價相對於當前價的分類：
      1: 漲超過 delta，-1: 跌超過 delta，0: 否則。
    """
    df = df.copy()
    df['future_price'] = df['price'].shift(-n_tick)
    df['price_diff'] = df['future_price'] - df['price']
    df.dropna(subset=['future_price'], inplace=True)

    def label_row(diff):
        if diff > delta:
            return 1
        elif diff < -delta:
            return -1
        else:
            return 0

    df['label'] = df['price_diff'].apply(label_row)
    return df.drop(columns=['future_price', 'price_diff'])
```

#### 3. 動態上下界

* 以近期波動率 (rolling\_std) 計算 Δ：

  ```python
  df['rolling_std'] = df['price'].rolling(window=20).std()
  df['delta'] = df['rolling_std'] * k  # k 為調整係數
  ```
* 若 `price.shift(-N) - price > delta` → label = 1；類似方式計算下跌 → -1；其餘 = 0。

#### 4. 多階段標註

1. **階段一：盤勢類別標註**

   * 標註當下 1 分鐘內是否 {微漲, 微跌, 盤整}。
2. **階段二：Tick 級價差標註**

   * 若階段一屬「趨勢」，再以未來 N tick 價差做 +1/0/-1 標註；若「盤整」，直接 label = 0。

---

### 2.B.3 時序資料切分 (Time-Series Split)

> \*\*目標：\*\*保留時間順序，避免未來資料洩漏到訓練集。

```python
from sklearn.model_selection import TimeSeriesSplit

def timeseries_split(df: pd.DataFrame, features: list, target: str, n_splits: int = 5, test_size: int = None):
    """
    將 df 以時間順序切分為多折 (train/test)，確保每一折的測試集在更接近當前時間的位置。
    """
    X = df[features]
    y = df[target]
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    splits = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        splits.append((X_train, X_test, y_train, y_test))
    return splits

# 使用範例
splits = timeseries_split(df, features, 'label', n_splits=10, test_size=500)
X_train, X_test, y_train, y_test = splits[-1]  # 取最後一折作為測試
```

#### 更進階切分策略

1. **跨週交叉驗證 (Cross-Week CV)**

   * 將一段長期資料 (如 1 個月) 拆成 N 個週 (Week1, Week2, …)。
   * 依序做 N 折 CV：Train=剩餘週集，Test=當週。
2. **滾動窗口訓練 (Rolling-Window Training)**

   ```
   Example:
   - 用最近 1 個月 Tick 資料訓練，隔天測試  
   - 隔天把最舊 1 天捨棄，加入最新一天，重新訓練  
   - 持續滾動，提升模型對市場漂移的適應力  
   ```
3. **Online Retraining（線上重訓）**

   * 每週或每天非交易時段自動拉取最新資料，使用滾動窗口更新訓練集後重訓，並自動部署新模型。

---

### 2.B.4 訓練與驗證 Pipeline 範例

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import joblib, os

def train_simple_rf(df, features, target, model_path):
    # 1. 時序切分 (取最後一折作為測試集)
    from utils.data_split import timeseries_split
    splits = timeseries_split(df, features, target, n_splits=5, test_size=None)
    X_train, X_test, y_train, y_test = splits[-1]

    # 2. 過採樣處理
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    # 3. 建立 Pipeline：先 Imputer 後 RandomForest
    imputer = SimpleImputer(strategy='mean')
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
    pipeline = Pipeline([('imputer', imputer), ('classifier', rf)])

    # 4. 訓練
    pipeline.fit(X_res, y_res)

    # 5. 評估
    from sklearn.metrics import classification_report
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    # 6. 儲存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"模型已儲存至：{model_path}")
```

* **流程說明**

  1. `timeseries_split` → 最後一折做測試
  2. `RandomOverSampler` → 訓練集少數類過採樣
  3. `Pipeline`： `SimpleImputer` + `RandomForestClassifier` (balanced\_subsample)
  4. 訓練、評估 (classification\_report)
  5. `joblib.dump` 儲存模型 (.pkl)

---

## ③ 第三階段：回測與績效驗證 (Backtesting & Performance Validation)

| 任務編號 | 任務內容                                  | 預期產出                                                                        | 涉及工具/模組                                                         | 備註         |
| ---- | ------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------- | ---------- |
| 3.1  | **簡單回測邏輯設計**                          | 回測偽代碼或 Python 函數原型，定義交易訊號、進出場條件、停損停利規則                                      | `backtest/core_logic.py`                                        | 考慮簡化交易成本模型 |
| 3.2  | **回測引擎初步實現**                          | 完成 `backtest/run_backtest.py`，可對測試集執行回測                                     | `pandas`, `numpy`                                               |            |
| 3.3  | **績效指標計算與分析**                         | 回測結果報告，包含 PnL (損益圖)、Win Rate (勝率)、Max Drawdown (最大回撤)、Sharpe Ratio (夏普比率) 等 | `pandas`, `numpy`, `matplotlib`, `utils/performance_metrics.py` |            |
| 3.4  | *(進階)* **Walk-Forward Validation 設計** | Walk-Forward 驗證流程設計與初步腳本，用以評估模型在不同時間段的穩定性                                   | —                                                               |            |
| 3.5  | *(進階)* **參數化策略回測**                    | 支援動態調整回測參數 (如停損點、進場閾值)，並比較不同參數組合結果                                          | —                                                               |            |

---

### 3.1 回測邏輯示例

```python
import pandas as pd
import numpy as np

def run_backtest(df, entry_signal_col='prediction', price_col='close',
                 stop_profit=10, stop_loss=10, cost_per_trade=2):
    """
    簡化版回測示例：
      - 當 entry_signal_col == 1 → 多單進場
      - 當 entry_signal_col == -1 → 空單進場
      - 設定停損 / 停利點
      - 計算每筆交易 PnL，扣除 cost_per_trade
    """
    balance = 0.0
    position = 0     # 1 = 多單, -1 = 空單, 0 = 空倉
    entry_price = 0.0
    results = []

    for idx, row in df.iterrows():
        price = row[price_col]
        signal = row[entry_signal_col]

        # 開倉
        if position == 0 and signal in [1, -1]:
            position = signal
            entry_price = price
            continue

        # 多單持有中
        if position == 1:
            if price >= entry_price + stop_profit or price <= entry_price - stop_loss:
                pnl = (price - entry_price) - cost_per_trade
                balance += pnl
                results.append({'exit_time': row['timestamp'], 'pnl': pnl})
                position = 0

        # 空單持有中
        if position == -1:
            if price <= entry_price - stop_profit or price >= entry_price + stop_loss:
                pnl = (entry_price - price) - cost_per_trade
                balance += pnl
                results.append({'exit_time': row['timestamp'], 'pnl': pnl})
                position = 0

    return pd.DataFrame(results), balance
```

**績效指標計算示例：**

```python
import pandas as pd
import numpy as np

def calculate_performance(trades: pd.DataFrame):
    """
    trades: DataFrame, 包含列 'pnl'
    返回 Victory Rate, Avg PnL, Max Drawdown, Sharpe Ratio
    """
    balance_curve = trades['pnl'].cumsum()
    total_trades = len(trades)
    win_trades = trades[trades['pnl'] > 0].shape[0]
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    avg_pnl = trades['pnl'].mean() if total_trades > 0 else 0

    # Max Drawdown
    roll_max = balance_curve.cummax()
    drawdown = (roll_max - balance_curve) / roll_max
    max_drawdown = drawdown.max() if not drawdown.empty else 0

    # Sharpe Ratio (risk-free rate 假設 0)
    returns = trades['pnl']
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0

    return {
        'Win Rate': win_rate,
        'Avg PnL': avg_pnl,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
    }
```

---

## ④ 第四階段：前端功能完善與即時化 (Frontend Enhancement & Real-time Integration)

| 任務編號 | 任務內容                | 預期產出                                                          | 涉及工具/模組                                    | 備註 |
| ---- | ------------------- | ------------------------------------------------------------- | ------------------------------------------ | -- |
| 4.1  | **整合進階模型預測**        | Streamlit 應用能載入並使用 ②.B 階段訓練的進階模型（如 LSTM/Transformer）          | `streamlit`                                |    |
| 4.2  | **前端可視化增強**         | Streamlit 頁面新增更豐富圖表：<br> - 原始價格＋技術指標疊加 <br> - 模型預測訊號 (買/賣點標記) | `streamlit`, `plotly` (推薦) / `matplotlib`  |    |
| 4.3  | **SHAP 解釋圖整合至前端**   | Streamlit 可顯示單筆預測的 SHAP 解釋圖或整體特徵重要性                           | `streamlit`, `shap`, `matplotlib`          |    |
| 4.4  | **預測紀錄與版本管理**       | 將預測結果 (含輸入特徵、模型版本) 儲存到 CSV 或 SQLite                           | `pandas`, `sqlite3`                        |    |
| 4.5  | *(進階)* **模擬即時資料串接** | 使用 `websocket-client` + `asyncio` 接收模擬 Tick Feed，觸發即時預測並更新前端  | `websocket-client`, `asyncio`, `streamlit` |    |

---

## ⑤ 第五階段：自動化與部署 (Automation & Deployment) *(選做/遠期)*

| 任務編號 | 任務內容                  | 預期產出                                       | 涉及工具/模組          | 備註 |
| ---- | --------------------- | ------------------------------------------ | ---------------- | -- |
| 5.1  | **Docker 化專案**        | `Dockerfile` 及相關配置，將專案打包為 Docker image     | `Docker`         |    |
| 5.2  | **CI/CD 或定時 Retrain** | GitHub Actions workflow 設定，用於自動化測試、建置與模型重訓 | `GitHub Actions` |    |
| 5.3  | **Paper Trading 模擬**  | 串接虛擬交易所 API 進行模擬下單與績效追蹤                    | 虛擬交易所 API        |    |

---

## 📊 模組互動示意圖 (Mermaid)

```mermaid
graph TD
    A[原始資料 CSV] --> B[資料準備與探索]
    B -- 清洗與基礎特徵 --> Proto_M[簡易模型訓練]
    Proto_M -- 簡易模型 --> Proto_API[簡易模型推論]
    A_upload[上傳 / 範例資料] --> Proto_API
    Proto_API -- 簡易預測 --> Proto_UI[Streamlit 快速原型 UI]

    B -- 完整特徵工程 --> C[完整特徵資料集]
    C -- 訓練 / 驗證 / 測試集 --> D[進階模型訓練模組]
    D -- 訓練好的模型 (.pkl/.json) --> E[模型推論 API (進階)]
    D -- 訓練好的模型 & 測試集 --> F[回測模組]
    C -- 測試集 --> F
    E -- 預測結果 & SHAP 值 --> G[Streamlit 前端 (完整功能)]
    F -- 回測績效報告 --> G
    H[新進 Tick/分K 資料（模擬或即時）] --> E

    subgraph Iteration_1_Rapid_Prototype
        A
        B
        Proto_M
        Proto_API
        A_upload
        Proto_UI
    end

    subgraph Iteration_2_and_Onwards
        C
        D
        F
        E
        G
        H
    end

    classDef data fill:#e6f2ff,stroke:#337ab7,stroke-width:2px;
    classDef core fill:#dff0d8,stroke:#3c763d,stroke-width:2px;
    classDef app fill:#fcf8e3,stroke:#8a6d3b,stroke-width:2px;
    classDef proto fill:#f5e6ff,stroke:#6a3ab2,stroke-width:2px;

    class A,H,A_upload data;
    class B,C,D,F core;
    class E,G app;
    class Proto_M,Proto_API,Proto_UI proto;
```

---

## 參考資料與工具推薦

1. **技術指標函式庫**

   * [ta-lib](https://mrjbq7.github.io/ta-lib/)：最常見的 C/C++ 指標套件 Python 版本
   * [pandas-ta](https://github.com/twopirllc/pandas-ta)：更易上手的 Python 技術指標套件
2. **資料切分與回測**

   * `sklearn.model_selection.TimeSeriesSplit`
   * 回測引擎：`backtrader`, `bt`, `zipline`
3. **深度學習時序模型**

   * `tensorflow.keras` / `pytorch` 中的 LSTM、GRU、Transformer
   * [GluonTS](https://ts.gluon.ai/)：AWS 推出的 Time Series 庫
4. **監控與部署**

   * [Docker 官方文件](https://docs.docker.com/)
   * [GitHub Actions 官方文件](https://docs.github.com/actions)

---

> **結語：**
> 這份 Side Project 文件涵蓋從資料準備、特徵工程、模型訓練、回測驗證，到前端 Streamlit 部署與自動化流程。
> 建議先快速完成「端到端原型」（②.A），確保資料流、特徵、訓練、預測、回測都能跑通，再逐步往「進階特徵與模型」、「回測強化」、「前端可視化」與「部署自動化」擴充。
> 如此能兼顧開發速度與後續可擴充性，並在市場實盤環境中持續迭代、優化。祝開發順利！

---

**檔案建議路徑：**

```
├── README.md              # 本份文件
├── data/                  # 原始與清洗後數據
│   ├── raw/               # Tick/分K 原始檔
│   └── processed/         # 經過清洗與分K後的資料
├── features/              # 特徵工程相關腳本
│   ├── feature_engineering.py
│   └── labeling.py
├── utils/                 # 通用工具
│   ├── data_cleaner.py
│   ├── data_split.py
│   └── performance_metrics.py
├── models/                # 訓練腳本與模型
│   ├── simple_model_trainer.py
│   ├── train_xgboost.py
│   └── trained_models/    # 儲存所有 .pkl 模型
├── backtest/              # 回測腳本
│   ├── core_logic.py
│   └── run_backtest.py
├── app/                   # Streamlit 前端
│   ├── main_app.py
│   ├── 1_train_model.py
│   ├── 2_predict_live.py
│   └── 3_backtest.py
└── notebooks/             # Jupyter Notebook
    ├── 01_eda.ipynb
    └── 02_shap_analysis.ipynb
```

---

**License:** MIT
**Author:** Yidti
**Contact:** bonjour.luc@gmail.com

EOF
