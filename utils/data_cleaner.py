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
    df = df[(df['close'] > 0) & (df['close'] < df['close'].quantile(0.99))]
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

    # 處理非交易時間資料 (如果需要)
    day_start = '08:45'
    day_end = '13:45'
    night_start = '15:00'
    night_end = '05:00'

    day_session = df.between_time(day_start, day_end)
    night_session = df.between_time(night_start, night_end)
    
    # 處理跨日夜盤情況
    # 如果 night_start > night_end，表示夜盤跨日
    if pd.to_datetime(night_start).time() > pd.to_datetime(night_end).time():
        # 分成兩段篩選：從 night_start 到午夜，以及從午夜到 night_end
        night_session_part1 = df.loc[df.index.time >= pd.to_datetime(night_start).time()]
        night_session_part2 = df.loc[df.index.time <= pd.to_datetime(night_end).time()]
        night_session = pd.concat([night_session_part1, night_session_part2])
    else:
        night_session = df.between_time(night_start, night_end)

    df = pd.concat([day_session, night_session]).sort_index()

    print("\n時間序列處理後：")
    print(df.head())
    print(df.index.dtype)
    print(f"資料筆數：{len(df)}")
    
    return df