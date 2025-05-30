import streamlit as st
import pandas as pd
import sys
import os
import joblib
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.simple_model_trainer import train_and_save_simple_model
from utils.data_cleaner import clean_data, process_time_series

def load_data(uploaded_file):
    ext = uploaded_file.name.split('.')[-1]
    if ext == 'csv':
        return pd.read_csv(uploaded_file)
    elif ext == 'parquet':
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError("不支援的檔案格式")

def get_model_name(uploaded_file):
    parts = uploaded_file.name.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3]) + '_model.pkl'
    return uploaded_file.name.split('.')[0] + '_model.pkl'

def main():
    st.set_page_config(page_title="金融期貨 Tick 資料預測原型", layout="wide")
    st.title("🚀 金融期貨 Tick 資料即時預測系統")

    features = ['close', 'volume', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume', 'tick_type', 'is_traffic_limited', 'SMA_20', 'EMA_20', 'RSI_14', 'BB_Middle', 'BB_Upper', 'BB_Lower']
    target = 'label'

    action = st.radio("選擇操作：", ["訓練新模型", "使用既有模型預測"], index=1)

    if action == "訓練新模型":
        st.header("1️⃣ 上傳訓練資料")
        train_file = st.file_uploader("請上傳 CSV 或 Parquet 檔案", type=["csv", "parquet"])

        if train_file:
            try:
                df = load_data(train_file)
                st.success("資料載入成功！")
                st.dataframe(df.head())

                if target not in df.columns:
                    st.error(f"資料缺少目標欄位 '{target}'")
                    return

                model_name = get_model_name(train_file)
                model_path = os.path.join('trained_models', model_name)

                train_and_save_simple_model(df, features, target, model_path)
                st.success(f"模型訓練完成並儲存為 {model_name}")
            except Exception as e:
                st.error(f"訓練錯誤: {e}")

    else:
        model_dir = 'trained_models'
        models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

        if not models:
            st.warning("沒有已儲存的模型，請先訓練一個")
            return

        model_file = st.selectbox("選擇模型：", models)
        if model_file:
            try:
                model_path = os.path.join(model_dir, model_file)
                model = joblib.load(model_path)
                st.success(f"已載入模型 {model_file}")
            except Exception as e:
                st.error(f"模型載入錯誤: {e}")
                return

            st.header("2️⃣ 上傳預測資料")
            pred_file = st.file_uploader("請上傳 CSV 或 Parquet 檔案", type=["csv", "parquet"])

            if pred_file:
                try:
                    df = load_data(pred_file)
                    st.success("資料載入成功！")
                    st.dataframe(df.head())

                    if 'ts' in df.columns:
                        df['timestamp'] = df['ts']
                        df.drop(columns=['ts'], inplace=True)

                    if 'timestamp' not in df.columns:
                        st.error("資料缺少 'timestamp' 欄位")
                        return

                    df = clean_data(df)
                    df = process_time_series(df)
                    df.reset_index(inplace=True)
                    st.success("資料預處理完成")
                    st.dataframe(df.head())

                    if all(f in df.columns for f in features):
                        predictions = model.predict(df[features])
                        df['prediction'] = predictions

                        st.subheader("📈 收盤價與預測標註圖")
                        fig = px.line(df, x='timestamp', y='close', title='收盤價與預測')

                        # 標註 -1 (紅色) 與 1 (綠色)
                        df_markers = df[df['prediction'].isin([-1, 1])]

                        df_markers['marker_symbol'] = df_markers['prediction'].map({1: 'triangle-up', -1: 'triangle-down'})

                        for pred_value, color, symbol in [(-1, 'red', 'triangle-down'), (1, 'green', 'triangle-up')]:
                            subset = df_markers[df_markers['prediction'] == pred_value]
                            fig.add_scatter(
                                x=subset['timestamp'],
                                y=subset['close'],
                                mode='markers',
                                marker=dict(
                                    color=color,
                                    size=10,
                                    symbol=symbol
                                ),
                                name=f'預測 {pred_value}'
                            )


                        # fig.add_scatter(
                        #     x=df_markers['timestamp'],
                        #     y=df_markers['close'],
                        #     mode='markers',
                        #     marker=dict(
                        #         color=df_markers['prediction'].map({1: 'green', -1: 'red'}),
                        #         size=8,
                        #         symbol='circle'
                        #     ),
                        #     name=f'預測 {pred_value}'
                        # )

                        st.plotly_chart(fig, use_container_width=True)

                        st.header("3️⃣ 下載預測結果")
                        st.download_button(
                            "下載預測結果 (CSV)",
                            df.to_csv(index=False).encode('utf-8'),
                            file_name="predicted_data.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("資料缺少部分必要特徵欄位")

                except Exception as e:
                    st.error(f"預測錯誤: {e}")

if __name__ == "__main__":
    main()
