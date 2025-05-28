import streamlit as st
import pandas as pd
import sys
import os
import joblib
import plotly.express as px # 導入 plotly.express

# 將專案根目錄添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.simple_model_trainer import train_and_save_simple_model
from utils.data_splitter import split_data
from utils.data_cleaner import clean_data, process_time_series # 導入資料清洗和時間序列處理函數

def main():
    st.set_page_config(page_title="金融期貨 Tick 資料預測原型", layout="wide")

    st.title("🚀 金融期貨 Tick 資料即時預測系統 - 快速原型")
    st.write("此應用展示了基於簡易模型的 Tick 資料預測流程。")

    st.header("1. 資料載入")
    uploaded_file = st.file_uploader("上傳您的 CSV 或 Parquet 檔案", type=["csv", "parquet"])

    df = None
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == "parquet":
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("不支援的檔案格式。請上傳 CSV 或 Parquet 檔案。")
                st.stop() # 停止執行，避免後續錯誤

            st.success("檔案上傳成功！")
            st.write("資料預覽:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"讀取檔案時發生錯誤: {e}")
            st.info("請確保您已安裝 'pyarrow' 和 'fastparquet' 以支援 Parquet 檔案。")
    else:
        st.info("請上傳 CSV 檔案，或使用內建範例資料進行演示。")
        # 使用範例資料
        example_data = {
            'timestamp': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-01 09:00:01', '2023-01-01 09:00:02', '2023-01-01 09:00:03', '2023-01-01 09:00:04']),
            'close': [100, 101, 102, 101, 102],
            'volume': [1000, 1200, 1100, 1300, 1050],
            'bid_price': [99, 100, 101, 100, 101],
            'bid_volume': [900, 1100, 1000, 1200, 950],
            'ask_price': [101, 102, 103, 102, 103],
            'ask_volume': [1100, 1300, 1200, 1400, 1150],
            'tick_type': [1, 2, 1, 2, 1],
            'is_traffic_limited': [0, 0, 0, 0, 0],
            'SMA_20': [100, 100.5, 101, 101.2, 101.5],
            'EMA_20': [100, 100.6, 101.1, 101.3, 101.6],
            'RSI_14': [50, 55, 60, 58, 62],
            'BB_Middle': [100, 100.5, 101, 101.2, 101.5],
            'BB_Upper': [101, 101.5, 102, 102.2, 102.5],
            'BB_Lower': [99, 99.5, 100, 100.2, 100.5],
            'label': [0, 1, -1, 1, 0] # 加入 -1 類別
        }
        df = pd.DataFrame(example_data)
        st.write("使用內建範例資料:")
        st.dataframe(df.head())

    if df is not None:
        st.header("1.5 資料預處理")
        try:
            # 確保 'timestamp' 欄位存在，如果不存在則嘗試從 'ts' 轉換
            if 'ts' in df.columns and 'timestamp' not in df.columns:
                df['timestamp'] = df['ts']
                df = df.drop(columns=['ts'])
            elif 'ts' in df.columns and 'timestamp' in df.columns:
                df = df.drop(columns=['ts']) # 如果兩者都存在，則移除 'ts'

            if 'timestamp' not in df.columns:
                st.error("資料中缺少 'timestamp' 欄位，無法進行時間序列處理。")
                st.stop()

            df = clean_data(df)
            df = process_time_series(df)
            # 重置索引，確保 'timestamp' 成為一個常規欄位
            df = df.reset_index()
            st.success("資料預處理完成！")
            st.write("預處理後資料預覽:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"資料預處理失敗: {e}")
            st.stop()

        # 添加價格走勢圖
        st.subheader("價格走勢圖")
        # 添加價格走勢圖
        st.subheader("價格走勢圖")
        # 確保 'timestamp' 和 'close' 欄位在預處理後仍然存在
        if 'timestamp' in df.columns and 'close' in df.columns:
            fig = px.line(df, x='timestamp', y='close', title='收盤價走勢')
            
            # 疊加預測為 1 的點位
            if 'prediction' in df.columns:
                # 篩選出預測為 1 的資料點
                predicted_up_points = df[df['prediction'] == 1]
                if not predicted_up_points.empty:
                    fig.add_scatter(x=predicted_up_points['timestamp'], y=predicted_up_points['close'],
                                    mode='markers', name='預測上漲點位',
                                    marker=dict(color='green', size=8, symbol='triangle-up')) # 將顏色改為綠色

                # 篩選出預測為 0 的資料點
                # 篩選出預測為 0 的資料點
                predicted_neutral_points = df[df['prediction'] == 0]
                if not predicted_neutral_points.empty:
                    fig.add_scatter(x=predicted_neutral_points['timestamp'], y=predicted_neutral_points['close'],
                                    mode='markers', name='預測不漲不跌點位',
                                    marker=dict(color='blue', size=8, symbol='circle')) # 使用藍色和圓形

                # 篩選出預測為 -1 的資料點
                predicted_down_points = df[df['prediction'] == -1]
                if not predicted_down_points.empty:
                    fig.add_scatter(x=predicted_down_points['timestamp'], y=predicted_down_points['close'],
                                    mode='markers', name='預測下跌點位',
                                    marker=dict(color='red', size=8, symbol='triangle-down')) # 使用紅色和向下三角形

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("資料中缺少 'timestamp' 或 'close' 欄位，無法繪製價格走勢圖。")
            st.stop() # 如果缺少關鍵欄位，則停止應用

        st.header("2. 模型預測")

        # 定義模型預期的特徵和目標
        features_for_prediction = ['close', 'volume', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'tick_type', 'is_traffic_limited', 'SMA_20', 'EMA_20', 'RSI_14', 'BB_Middle', 'BB_Upper', 'BB_Lower']
        target_label = 'label'

        # 檢查資料是否包含所有必要的特徵欄位
        missing_features = [f for f in features_for_prediction if f not in df.columns]
        if missing_features:
            st.warning(f"上傳的資料缺少模型所需的特徵欄位: {', '.join(missing_features)}。將使用範例資料進行模型訓練和預測。")
            # 如果缺少特徵，則使用範例資料進行訓練和預測
            example_data_for_training = {
                'timestamp': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-01 09:00:01', '2023-01-01 09:00:02', '2023-01-01 09:00:03', '2023-01-01 09:00:04', '2023-01-01 09:00:05', '2023-01-01 09:00:06', '2023-01-01 09:00:07', '2023-01-01 09:00:08', '2023-01-01 09:00:09', '2023-01-01 09:00:10', '2023-01-01 09:00:11', '2023-01-01 09:00:12', '2023-01-01 09:00:13', '2023-01-01 09:00:14']),
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
                'label': [0, 1, -1, 1, 0, 1, -1, 1, 0, 1, -1, 1, 0, 1, -1] # 加入 -1 類別
            }
            df_for_prediction = pd.DataFrame(example_data_for_training)
            st.write("已切換至使用內建範例資料進行模型預測。")
        else:
            df_for_prediction = df.copy()

        model_path = 'models/simple_model.pkl'

        st.subheader("模型操作")
        model_option = st.radio(
            "選擇模型操作：",
            ("訓練新模型", "使用既有模型"),
            index=1 if os.path.exists(model_path) else 0 # 如果模型存在，預設為使用既有模型
        )

        model = None
        if model_option == "訓練新模型":
            st.info("正在訓練一個新的簡易模型...")
            # 確保用於訓練的資料包含目標欄位
            if target_label not in df_for_prediction.columns:
                # 為了演示，如果範例資料沒有 target，則根據 close 欄位變化自動生成 label
                # 1: 上漲, -1: 下跌, 0: 不變
                diff = df_for_prediction['close'].diff().fillna(0)
                df_for_prediction[target_label] = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                st.warning(f"範例資料中缺少 '{target_label}' 欄位，已根據收盤價變化自動生成用於模型訓練。")

            try:
                train_and_save_simple_model(df_for_prediction, features_for_prediction, target_label, model_path)
                st.success("簡易模型訓練並儲存完成！")
                model = joblib.load(model_path) # 訓練後立即載入新模型
            except Exception as e:
                st.error(f"模型訓練失敗: {e}")
                st.stop() # 訓練失敗則停止 Streamlit 應用
        elif model_option == "使用既有模型":
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    st.success("既有模型載入成功！")
                except Exception as e:
                    st.error(f"載入既有模型失敗: {e}")
                    st.stop() # 載入失敗則停止 Streamlit 應用
            else:
                st.warning("模型檔案不存在，請選擇 '訓練新模型' 或上傳資料以訓練模型。")
                st.stop() # 模型不存在則停止 Streamlit 應用

        if st.button("執行預測"):
            if df_for_prediction is not None and not df_for_prediction.empty:
                try:
                    # 確保預測資料只包含模型訓練時的特徵
                    X_predict = df_for_prediction[features_for_prediction]
                    predictions = model.predict(X_predict)
                    prediction_proba = model.predict_proba(X_predict)

                    df_for_prediction['prediction'] = predictions
                    df_for_prediction['prediction_proba_0'] = prediction_proba[:, 0]
                    df_for_prediction['prediction_proba_1'] = prediction_proba[:, 1]

                    st.write("預測結果:")
                    st.dataframe(df_for_prediction[['timestamp'] + features_for_prediction + ['prediction', 'prediction_proba_1']].tail()) # 顯示最後幾筆預測結果

                    st.subheader("預測結果摘要")
                    pred_counts = pd.Series(predictions).value_counts().sort_index() # 排序以確保順序
                    st.write(f"預測為 -1 (下跌) 的筆數: {pred_counts.get(-1, 0)}")
                    st.write(f"預測為 0 (不漲不跌) 的筆數: {pred_counts.get(0, 0)}")
                    st.write(f"預測為 1 (上漲) 的筆數: {pred_counts.get(1, 0)}")

                except Exception as e:
                    st.error(f"執行預測時發生錯誤: {e}")
            else:
                st.warning("沒有可用的資料進行預測。")

        st.header("3. 資料匯出")
        if df_for_prediction is not None and not df_for_prediction.empty:
            # 將 DataFrame 轉換為 Parquet 格式的 Bytes
            parquet_file = df_for_prediction.to_parquet(index=False)
            st.download_button(
                label="下載預測結果為 Parquet",
                data=parquet_file,
                file_name="predicted_data.parquet",
                mime="application/octet-stream"
            )
        else:
            st.info("沒有可匯出的資料。請先上傳資料並執行預測。")

if __name__ == "__main__":
    main()