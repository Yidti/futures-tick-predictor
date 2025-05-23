import pandas as pd

def create_up_down_label(
    data: pd.Series,
    window: str,
    threshold: float
) -> pd.Series:
    """
    針對 Tick 級別的收盤價 (data)，計算「往後 window 時間點的最近成交價」，
    並依照 price_change > threshold 來標記 (1: 上漲, 0: 下跌/持平)。

    Args:
        data: pandas.Series，index 必須是 DatetimeIndex，內容是收盤價 (price)。
        window: 字串（e.g., '30min' 或 '30T'），表示往後看的時間。
        threshold: 漲跌幅閾值，例如 0.001 代表 0.1%。

    Returns:
        pandas.Series，index 與 data 相同，元素為 1 或 0，代表未來 price change 是否大於 threshold。
        如果「未來時間點找不到對應成交」，該位置會是 NaN。
    """
    # 1. 確保 index 是 DatetimeIndex，並排序
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index 必須是 DatetimeIndex")
    data = data.sort_index()

    # 2. 把原始 Series 轉成 DataFrame，並 reset_index()
    df_orig = data.to_frame(name='price').reset_index()
    df_orig = df_orig.rename(columns={df_orig.columns[0]: 'time'})

    # 3. 計算「future_time」 = 當前時間 + window
    delta = pd.Timedelta(window)
    df_orig['future_time'] = df_orig['time'] + delta

    # 4. 準備「merge_asof」的基準表
    df_base = data.to_frame(name='future_price').reset_index()
    df_base = df_base.rename(columns={df_base.columns[0]: 'time'})

    # 5. merge_asof：左表(df_orig，key=future_time)，右表(df_base，key=time)
    df_merged = pd.merge_asof(
        df_orig.sort_values('future_time'),
        df_base.sort_values('time'),
        left_on='future_time',
        right_on='time',
        direction='forward'
    )

    # 6. 重新命名時間欄位：把 merge_asof 自動產生的 time_x 改回 time
    df_merged = df_merged.rename(columns={'time_x': 'time'})
    # （可選）如果想把 matched 未來時間保留，也可以改 time_y：
    # df_merged = df_merged.rename(columns={'time_y': 'matched_time'})

    # 7. 計算漲跌幅，並標記 0/1
    df_merged['price_change'] = (df_merged['future_price'] - df_merged['price']) / df_merged['price']
    df_merged['label'] = (df_merged['price_change'] > threshold).astype(int)

    # 8. 把 time 設為 index，回傳 label Series
    result = df_merged.set_index('time')['label']
    return result
