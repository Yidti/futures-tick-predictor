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

def calculate_macd(data: pd.Series, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """
    計算移動平均收斂散度 (Moving Average Convergence Divergence, MACD)。

    Args:
        data: 輸入的 pandas Series (e.g., 收盤價)。
        fast_window: 計算快速 EMA 的窗口大小 (預設為 12)。
        slow_window: 計算慢速 EMA 的窗口大小 (預設為 26)。
        signal_window: 計算訊號線 EMA 的窗口大小 (預設為 9)。

    Returns:
        包含 MACD 線、訊號線和柱狀圖的 pandas DataFrame。
    """
    ema_fast = data.ewm(span=fast_window, adjust=False).mean()
    ema_slow = data.ewm(span=slow_window, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'MACD_Line': macd_line,
        'Signal_Line': signal_line,
        'MACD_Histogram': histogram
    })

def calculate_kd(data: pd.DataFrame, window: int = 9) -> pd.DataFrame:
    """
    計算隨機指標 (KDJ)。

    Args:
        data: 包含 High, Low, Close 欄位的 pandas DataFrame。
        window: 計算窗口大小 (預設為 9)。

    Returns:
        包含 K 值和 D 值的 pandas DataFrame。
    """
    low_list = data['Low'].rolling(window=window).min()
    high_list = data['High'].rolling(window=window).max()

    rsv = ((data['Close'] - low_list) / (high_list - low_list)) * 100

    k = rsv.ewm(span=3, adjust=False).mean()
    d = k.ewm(span=3, adjust=False).mean()

    return pd.DataFrame({
        'K': k,
        'D': d
    })