import numpy as np
import pandas as pd


import numpy as np
import pandas as pd

def label_future_returns_fast(close_series: pd.Series, future_window='30min', pos_threshold=0.003, neg_threshold=-0.003) -> pd.Series:
    """
    快速標註：用未來固定時間窗口最後一筆價格計算報酬，並根據閾值標註多空持平。

    Args:
        close_series (pd.Series): 時間序列的收盤價，index 為時間戳。
        future_window (str): 代表未來時間窗口長度，如 '30min'、'1H' 等。
        pos_threshold (float): 上漲閾值，報酬大於等於此值標為 1。
        neg_threshold (float): 下跌閾值，報酬小於等於此值標為 -1。

    Returns:
        pd.Series: 與 close_series 對齊的標籤序列，值為 -1, 0, 1。
    """
    close_series = close_series.copy()
    close_series.index = pd.to_datetime(close_series.index).sort_values()
    
    df = pd.DataFrame({'close': close_series})
    
    freq = close_series.index.to_series().diff().median()
    if pd.isna(freq) or freq <= pd.Timedelta(0):
        raise ValueError("時間索引頻率無法計算，請確認時間索引是否正確且有序。")
    
    steps = int(pd.Timedelta(future_window) / freq)
    if steps <= 0:
        raise ValueError(f"計算出的步數為非正值 steps={steps}，請檢查 future_window 與資料頻率。")

    df['future_return'] = df['close'].shift(-steps) / df['close'] - 1

    df['label'] = 0
    df.loc[df['future_return'] >= pos_threshold, 'label'] = 1
    df.loc[df['future_return'] <= neg_threshold, 'label'] = -1

    # 尾端 NaN 標籤可視需求填 0 或丟棄
    df['label'] = df['label'].fillna(0).astype(int)

    return df['label']

    """
    快速標註：用未來固定窗口最後一筆價格計算報酬
    """
    close_series = close_series.sort_index()
    close_series.index = pd.to_datetime(close_series.index)

    df = pd.DataFrame({'close': close_series})

    # 計算 future window 對應的 shift 步數
    freq = close_series.index.to_series().diff().median()
    steps = int(pd.Timedelta(future_window) / freq)

    # 快速計算未來報酬
    df['future_return'] = df['close'].shift(-steps) / df['close'] - 1

    # 標註
    df['label'] = 0
    df.loc[df['future_return'] >= pos_threshold, 'label'] = 1
    df.loc[df['future_return'] <= neg_threshold, 'label'] = -1

    labels = df['label'].astype(int)
    return labels