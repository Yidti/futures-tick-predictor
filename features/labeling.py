import numpy as np
import pandas as pd


import numpy as np
import pandas as pd

def label_future_returns_fast(close_series: pd.Series, future_window='30min', pos_threshold=0.003, neg_threshold=-0.003) -> pd.Series:
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