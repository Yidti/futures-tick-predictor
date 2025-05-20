import pandas as pd

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    計算簡單移動平均 (Simple Moving Average, SMA)。

    Args:
        data: 輸入的 pandas Series (e.g., 收盤價)。
        window: 計算 SMA 的窗口大小。

    Returns:
        計算出的 SMA pandas Series。
    """
    return data.rolling(window=window).mean()

# TODO: 添加其他技術指標計算函數