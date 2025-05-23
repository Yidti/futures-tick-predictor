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

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """
    計算指數移動平均 (Exponential Moving Average, EMA)。

    Args:
        data: 輸入的 pandas Series (e.g., 收盤價)。
        window: 計算 EMA 的窗口大小。

    Returns:
        計算出的 EMA pandas Series。
    """
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    計算相對強弱指數 (Relative Strength Index, RSI)。

    Args:
        data: 輸入的 pandas Series (e.g., 收盤價)。
        window: 計算 RSI 的窗口大小 (預設為 14)。

    Returns:
        計算出的 RSI pandas Series。
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std_dev: int = 2) -> pd.DataFrame:
    """
    計算布林通道 (Bollinger Bands, BB)。

    Args:
        data: 輸入的 pandas Series (e.g., 收盤價)。
        window: 計算移動平均和標準差的窗口大小 (預設為 20)。
        num_std_dev: 標準差的倍數 (預設為 2)。

    Returns:
        包含中軌、上軌和下軌的 pandas DataFrame。
    """
    sma = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()

    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)

    return pd.DataFrame({
        'BB_Middle': sma,
        'BB_Upper': upper_band,
        'BB_Lower': lower_band
    })