# 🚀 金融期貨 Tick 資料即時預測系統 - 詳細執行計畫 (調整版) 🚀

**專案總目標：** 建立一個能針對台指期等 Tick / 分K 資料進行 **即時特徵工程 + 預測 + 回測驗證** 的 **可視化預測系統**，可部署為 Streamlit 應用，未來可銜接 WebSocket 即時資料與真實交易 API。

**核心技術棧 (初步建議)：**
*   **資料處理與分析：** `pandas`, `numpy`
*   **技術指標：** `ta-lib`
*   **機器學習模型：** `scikit-learn`, `xgboost` (初期), `tensorflow/keras` 或 `pytorch` (進階 LSTM/Transformer)
*   **模型解釋性：** `shap`
*   **回測框架：** 初期可自建簡化版，或考慮 `backtrader`, `bt`
*   **前端與可視化：** `streamlit`, `plotly`, `matplotlib`
*   **版本控制：** `Git`
*   **開發環境：** `Jupyter Notebook` (用於 EDA 與實驗), Python IDE

---

## ⓪ 專案初始化與環境設定

| 任務編號 | 任務內容 | 預期產出 | 涉及工具/模組 | 備註 |
|---|---|---|---|---|
| 0.1 | 建立專案根目錄及子目錄結構 | 符合先前討論的專案目錄結構 (`data/`, `features/`, `models/`, `backtest/`, `app/`, `notebooks/`, `utils/`) | 檔案系統 |  |
| 0.2 | 初始化 Git 版本控制 | `.git` 目錄，專案納入版控 | `git` |  |
| 0.3 | 建立 `requirements.txt` | `requirements.txt` 檔案，包含核心依賴套件的初步列表 | 文字編輯器 | 例如：`pandas`, `numpy`, `scikit-learn`, `xgboost`, `ta-lib`, `streamlit`, `jupyterlab`, `matplotlib`, `plotly`, `shap` |
| 0.4 | 設定 Python 虛擬環境 | 啟用的虛擬環境 (e.g., venv, conda) | `venv` / `conda` | 強烈建議，以隔離專案依賴 |
| 0.5 | 安裝初步依賴套件 | 虛擬環境中已安裝 `requirements.txt` 所列套件 | `pip` | `pip install -r requirements.txt` |
| 0.6 | 設定 Jupyter Notebook/Lab 環境 | 可啟動並運行的 Jupyter 環境 | `jupyterlab` / `notebook` | 用於後續的 EDA 和實驗 |

---

## ① 第一階段：資料準備與探索 (Data Preparation & EDA)

| 任務編號 | 任務內容 | 預期產出 | 涉及工具/模組 | 備註 |
|---|---|---|---|---|
| 1.1 | **資料載入與初步檢視** | Python 腳本 (`.py` 或 `.ipynb`) 可成功讀取提供的 CSV 歷史資料 | `pandas`, `notebooks/` | 確認欄位名稱、資料型態、資料筆數、時間範圍 |
| 1.2 | **資料清洗** | 清洗後的 DataFrame：處理缺失值 (NaN)、異常價格 (outliers)、重複資料 | `pandas`, `numpy`, `notebooks/` | 記錄清洗策略與過程 |
| 1.3 | **時間序列處理** | DataFrame 以時間欄位為索引，時間格式標準化，處理非交易時間資料 (若適用) | `pandas`, `notebooks/` |  |
| 1.4 | **初步探索性資料分析 (EDA)** | Jupyter Notebook (`notebooks/01_eda.ipynb`) 包含：<br> - 價格 (開高低收) 走勢圖 <br> - 成交量變化圖 <br> - 基礎波動率分析 (e.g., 日報酬率標準差) <br> - 價格與成交量分佈圖 | `pandas`, `matplotlib`, `seaborn`, `plotly` (可選) |  |
| 1.5 | **技術指標特徵工程 (基礎)** | DataFrame 新增數個核心技術指標欄位 (e.g., SMA, EMA, RSI) | `ta-lib`, `pandas`, `features/feature_engineering.py` (初步) | 選擇少量、易於理解的指標用於快速原型 |
| 1.6 | **製作標註欄位 (Target Labeling)** | DataFrame 新增目標預測欄位 (e.g., `target_price_up_N_ticks`)，值為 0/1 | `pandas`, `numpy`, `features/labeling.py` (初步) | 明確定義預測目標 (e.g., 未來 N Tick/分鐘 後價格漲跌幅是否超過 X%) |

---

## ②.A 快速原型：簡易模型與前端展示 (Rapid Prototype: Simple Model & Frontend)

| 任務編號 | 任務內容 | 預期產出 | 涉及工具/模組 | 備註 |
|---|---|---|---|---|
| 2.A.1 | **資料切分 (簡易)** | 函數將資料切分為訓練集和測試集 (用於原型) | `scikit-learn` |  |
| 2.A.2 | **訓練極簡模型** | 訓練一個非常簡單的分類模型 (e.g., Logistic Regression 或基於少量特徵的 Dummy Classifier/XGBoost) 並儲存模型檔案 | `scikit-learn`/`xgboost`, `joblib`/`pickle` | 目標是跑通流程，非追求高精度 |
| 2.A.3 | **Streamlit UI 基礎架構** | `app/main_app.py` 雛形，包含：<br> - 頁面標題與基本佈局 <br> - CSV 資料上傳功能 (或使用內建範例資料) | `streamlit` |  |
| 2.A.4 | **簡易模型推論整合** | Streamlit 應用能載入訓練好的極簡模型，對上傳/範例資料進行預測 | `streamlit`, `joblib`/`pickle` |  |
| 2.A.5 | **前端結果初步展示** | Streamlit 頁面能簡單展示：<br> - 輸入資料的摘要或圖表 <br> - 模型預測結果 (e.g., 預測為上漲/下跌) | `streamlit`, `pandas`, `matplotlib`/`plotly` (可選) | 讓使用者能看到一個可互動的輸入輸出流程 |

---

## ②.B 第二階段：模型訓練與預測模組 (進階) (Model Training & Prediction - Advanced)

| 任務編號 | 任務內容 | 預期產出 | 涉及工具/模組 | 備註 |
|---|---|---|---|---|
| 2.B.1 | **進階特徵工程** | 擴展 `features/feature_engineering.py`，加入更多技術指標 (MACD, KD, Bollinger Bands 等) 及其他潛在特徵 | `ta-lib`, `pandas` |  |
| 2.B.2 | **時間序列資料切分策略** | Python 函數/類別，用於將時間序列資料切分為訓練集、驗證集、測試集 (e.g., `TimeSeriesSplit`) | `scikit-learn`, `utils/data_split.py` | 確保時間順序性，避免資料洩漏 |
| 2.B.3 | **訓練 XGBoost 模型** | 訓練好的 XGBoost 模型檔案；較完善的訓練腳本 `models/train_xgboost.py` | `xgboost`, `scikit-learn`, `pandas` |  |
| 2.B.4 | **模型評估指標選擇與實現** | Python 函數，用於計算 Accuracy, Precision, Recall, F1-score, ROC-AUC 等指標 | `scikit-learn.metrics`, `utils/evaluation.py` |  |
| 2.B.5 | **訓練流程腳本化** | `models/train.py` (或擴展 `train_xgboost.py`)，能整合資料載入、特徵工程、資料切分、模型訓練、評估及儲存 | Python scripting |  |
| 2.B.6 | **模型解釋性 (SHAP)** | Jupyter Notebook (`notebooks/02_shap_analysis.ipynb`) 展示 SHAP 值圖表及特徵重要性分析 | `shap`, `xgboost`, `matplotlib` |  |
| 2.B.7 | (選做) **嘗試 LSTM/Transformer 模型** | 初步的 LSTM/Transformer 模型架構與訓練腳本 | `tensorflow/keras` or `pytorch` |  |

---

## ③ 第三階段：回測與績效驗證 (Backtesting & Performance Validation)

| 任務編號 | 任務內容 | 預期產出 | 涉及工具/模組 | 備註 |
|---|---|---|---|---|
| 3.1 | **簡單回測邏輯設計** | 回測邏輯的偽代碼或 Python 函數原型，定義交易訊號、進出場條件、停損停利規則 | `backtest/core_logic.py` | 考慮簡化的交易成本模型 |
| 3.2 | **回測引擎初步實現** | Python 腳本 `backtest/run_backtest.py`，能在測試集上根據模型預測執行回測 | `pandas`, `numpy` |  |
| 3.3 | **績效指標計算與分析** | 回測結果報告，包含 PnL (損益圖)、Win Rate (勝率)、Max Drawdown (最大回撤)、Sharpe Ratio (夏普比率) 等 | `pandas`, `numpy`, `matplotlib`, `utils/performance_metrics.py` |  |
| 3.4 | (進階) **Walk-Forward Validation 設計** | Walk-forward 驗證的流程設計與初步腳本 |  | 評估模型在不同時間段的穩定性 |
| 3.5 | (進階) **參數化策略回測** | 支援調整回測參數 (如停損點、進場閾值) 並比較結果的機制 |  |  |

---

## ④ 第四階段：前端功能完善與即時化 (Frontend Enhancement & Real-time Integration)

| 任務編號 | 任務內容 | 預期產出 | 涉及工具/模組 | 備註 |
|---|---|---|---|---|
| 4.1 | **整合進階模型預測** | Streamlit 應用能載入並使用 ②.B 階段訓練的較複雜模型進行預測 | `streamlit` |  |
| 4.2 | **前端結果可視化 (增強)** | Streamlit 頁面展示更豐富的圖表：<br> - 原始價格圖與技術指標疊加 <br> - 模型預測訊號 (買/賣點標記在價格圖上) | `streamlit`, `plotly` (推薦) / `matplotlib` |  |
| 4.3 | **SHAP 解釋圖整合至前端** | Streamlit 頁面能顯示單筆預測的 SHAP 解釋圖或整體特徵重要性 | `streamlit`, `shap`, `matplotlib` |  |
| 4.4 | **預測紀錄儲存** | 將預測結果 (及相關輸入特徵、模型版本) 儲存至 CSV 或 SQLite | `pandas`, `sqlite3` |  |
| 4.5 | (進階) **串接模擬即時資料源** | 嘗試用 `websocket-client` 和 `asyncio` 接收模擬的 Tick Feed 並觸發預測，更新前端顯示 | `websocket-client`, `asyncio`, `streamlit` |  |

---

## ⑤ 第五階段：自動化與部署 (Automation & Deployment) - (選做/遠期)

| 任務編號 | 任務內容 | 預期產出 | 涉及工具/模組 | 備註 |
|---|---|---|---|---|
| 5.1 | Docker 化專案 | `Dockerfile` 及相關配置，使專案可打包為 Docker image | `Docker` |  |
| 5.2 | CI/CD 或定時 Retrain | GitHub Actions 或類似工具的 workflow 設定，用於自動化測試、建置、模型重訓 | `GitHub Actions` |  |
| 5.3 | Paper Trading 模擬 | 串接虛擬交易 API 進行模擬下單與績效追蹤 | 虛擬交易所 API |  |

---

## 📊 模組互動示意圖 (Mermaid) - 強調迭代

```mermaid
graph TD
    A[原始資料 .csv] --> B[資料準備與探索];
    B -- 清洗與基礎特徵 --> Proto_M[簡易模型訓練];
    Proto_M -- 簡易模型 --> Proto_API[簡易模型推論];
    A_upload[上傳 / 範例資料] --> Proto_API;
    Proto_API -- 簡易預測 --> Proto_UI[Streamlit 快速原型 UI];

    B -- 完整特徵工程 --> C[完整特徵資料集];
    C -- 訓練 / 驗證 / 測試集 --> D[進階模型訓練模組];
    D -- 訓練好的模型 (.pkl / .json) --> E[模型推論 API - 進階];
    D -- 訓練好的模型 & 測試集 --> F[回測模組];
    C -- 測試集 --> F;
    E -- 預測結果 & SHAP 值 --> G[Streamlit 前端應用 - 完整功能];
    F -- 回測績效報告 --> G;
    H[新進 Tick / 分K 資料（模擬或即時）] --> E;

    subgraph Iteration_1_Rapid_Prototype
        A
        B
        Proto_M
        Proto_API
        A_upload
        Proto_UI
    end

    subgraph Iteration_2_onwards_Enhancement_and_Full_Features
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
