# 🚀 金融期貨 Tick 資料即時預測系統

**專案總目標：** 建立一個能針對台指期等 Tick / 分K 資料進行 **即時特徵工程 + 預測 + 回測驗證** 的 **可視化預測系統**，可部署為 Streamlit 應用，未來可銜接 WebSocket 即時資料與真實交易 API。

## 核心技術棧

*   **資料處理與分析：** `pandas`, `numpy`
*   **技術指標：** `ta-lib`
*   **機器學習模型：** `scikit-learn`, `xgboost` (初期), `tensorflow/keras` 或 `pytorch` (進階 LSTM/Transformer)
*   **模型解釋性：** `shap`
*   **回測框架：** 初期可自建簡化版，或考慮 `backtrader`, `bt`
*   **前端與可視化：** `streamlit`, `plotly`, `matplotlib`
*   **版本控制：** `Git`
*   **開發環境：** `Jupyter Notebook` (用於 EDA 與實驗), Python IDE

## 專案階段概述

本專案分為以下幾個主要階段：

1.  **第一階段：資料準備與探索 (Data Preparation & EDA)**
    *   資料載入與初步檢視
    *   資料清洗
    *   時間序列處理
    *   初步探索性資料分析 (EDA)
    *   技術指標特徵工程 (基礎)
    *   製作標註欄位 (Target Labeling)

2.  **快速原型：簡易模型與前端展示 (Rapid Prototype: Simple Model & Frontend)**
    *   資料切分 (簡易)
    *   訓練極簡模型
    *   Streamlit UI 基礎架構
    *   簡易模型推論整合
    *   前端結果初步展示

3.  **第二階段：模型訓練與預測模組 (進階) (Model Training & Prediction - Advanced)**
    *   進階特徵工程
    *   時間序列資料切分策略
    *   訓練 XGBoost 模型
    *   模型評估指標選擇與實現
    *   訓練流程腳本化
    *   模型解釋性 (SHAP)
    *   (選做) 嘗試 LSTM/Transformer 模型

4.  **第三階段：回測與績效驗證 (Backtesting & Performance Validation)**
    *   簡單回測邏輯設計
    *   回測引擎初步實現
    *   績效指標計算與分析
    *   (進階) Walk-Forward Validation 設計
    *   (進階) 參數化策略回測

5.  **第四階段：前端功能完善與即時化 (Frontend Enhancement & Real-time Integration)**
    *   整合進階模型預測
    *   前端結果可視化 (增強)
    *   SHAP 解釋圖整合至前端
    *   預測紀錄儲存
    *   (進階) 串接模擬即時資料源

6.  **第五階段：自動化與部署 (Automation & Deployment) - (選做/遠期)**
    *   Docker 化專案
    *   CI/CD 或定時 Retrain
    *   Paper Trading 模擬

## 如何開始

1.  **環境設定：** 確保已安裝 Python 3.x 及必要的套件 (參考 `requirements.txt`)。建議使用虛擬環境。
2.  **資料準備：** 準備您的歷史 Tick/分K 資料 (CSV 格式)。
3.  **執行 Notebooks：** 按照 `notebooks/` 目錄下的 Notebook 逐步進行資料處理和分析。
4.  **運行 Streamlit 應用：** 在完成模型訓練後，可以運行 `app/main_app.py` 來啟動可視化預測系統。

```bash
streamlit run app/main_app.py
```

**注意：** 本專案仍在開發階段，部分功能可能尚未完全實現。