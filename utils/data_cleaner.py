import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    處理資料中的缺失值、異常值和重複資料。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。

    Returns:
        pd.DataFrame: 清洗後的資料 DataFrame。
    """
    print("\n處理缺失值前：")
    print(df.isnull().sum())
    df = df.dropna() # 範例：簡單刪除含有缺失值的列
    print("\n處理缺失值後：")
    print(df.isnull().sum())

    print("\n處理異常值前：")
    print(df.describe())
    # 範例：移除價格為 0 或極端值的資料
    # df = df[(df['close'] > 0) & (df['close'] < df['close'].quantile(0.99))]
    print("\n處理異常值後：")
    print(df.describe())

    print("\n處理重複資料前：")
    print(f"重複資料筆數：{df.duplicated().sum()}")
    df = df.drop_duplicates(subset=['timestamp'])
    print("\n處理重複資料後：")
    print(f"重複資料筆數：{df.duplicated().sum()}")
    
    return df

def process_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    將時間欄位設定為索引，標準化時間格式，並處理非交易時間資料。

    Args:
        df (pd.DataFrame): 原始資料 DataFrame。

    Returns:
        pd.DataFrame: 處理時間序列後的資料 DataFrame。
    """
    # 將時間欄位設定為索引
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 標準化時間格式 (如果需要)
    df = df.sort_index()

    # 處理非交易時間資料
    day_start = '08:45'
    day_end = '13:45'
    night_start = '15:00'
    night_end = '05:00'

    day_start_time = pd.to_datetime(day_start).time()
    day_end_time = pd.to_datetime(day_end).time()
    night_start_time = pd.to_datetime(night_start).time()
    night_end_time = pd.to_datetime(night_end).time()

    cleaned_df = pd.DataFrame()
    unique_dates = df.index.normalize().unique()

    for current_date in unique_dates:
        # 日盤時間範圍
        day_session_start_dt = current_date.replace(hour=day_start_time.hour, minute=day_start_time.minute, second=day_start_time.second)
        day_session_end_dt = current_date.replace(hour=day_end_time.hour, minute=day_end_time.minute, second=day_end_time.second)
        
        # 夜盤時間範圍 (跨日)
        night_session_start_dt = current_date.replace(hour=night_start_time.hour, minute=night_start_time.minute, second=night_start_time.second)
        night_session_end_dt = (current_date + pd.Timedelta(days=1)).replace(hour=night_end_time.hour, minute=night_end_time.minute, second=night_end_time.second)
        
        # 篩選日盤數據
        day_data = df.loc[(df.index >= day_session_start_dt) & (df.index < day_session_end_dt)]
        
        # 篩選夜盤數據
        night_data_part1 = df.loc[(df.index >= night_session_start_dt) & (df.index <= current_date.replace(hour=23, minute=59, second=59))]
        night_data_part2 = df.loc[(df.index >= current_date.replace(hour=0, minute=0, second=0) + pd.Timedelta(days=1)) & (df.index < night_session_end_dt)]
        
        night_data = pd.concat([night_data_part1, night_data_part2])
        
        # 合併日盤和夜盤數據
        daily_trade_data = pd.concat([day_data, night_data])
        
        cleaned_df = pd.concat([cleaned_df, daily_trade_data])
            
    df = cleaned_df.sort_index()

    print("\n時間序列處理後：")
    print(df.head())
    print(df.index.dtype)
    print(f"資料筆數：{len(df)}")
    
    return df