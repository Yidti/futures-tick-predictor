import pandas as pd
import numpy as np

def create_up_down_label(data: pd.Series, window: str, threshold: float) -> pd.Series:
    """
    根據未來價格變化創建漲跌標籤。

    Args:
        data: 輸入的 pandas Series (e.g., 收盤價)。
        window: 向前看的時間窗口 (e.g., '5T' for 5 minutes)。
        threshold: 漲跌幅閾值。

    Returns:
        包含漲跌標籤 (1: 上漲, 0: 下跌/持平) 的 pandas Series。
    """
    future_price = data.shift(-pd.Timedelta(window))
    price_change = (future_price - data) / data
    label = (price_change > threshold).astype(int)
    return label

# TODO: 添加其他標註函數