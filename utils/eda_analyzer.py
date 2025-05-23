import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

def plot_daily_close_price(df: pd.DataFrame):
    """
    繪製每日收盤價互動式走勢圖 (日盤/夜盤區分)，帶有日期範圍選擇器。
    線段只在有實際交易數據的點上繪製，沒有交易則斷開。

    Args:
        df (pd.DataFrame): 包含 'close' 欄位的 DataFrame，索引為時間。
    """
    start_time = df.index.min().floor('min')
    end_time = df.index.max().ceil('min')
    
    # Create a full minute-by-minute index for the entire period
    full_minute_index = pd.date_range(start=start_time, end=end_time, freq='min')
    
    # Ensure the original 'close' data is at minute frequency before reindexing
    # Use .last() to pick the last observed price in each minute interval
    # This handles cases where original data might be tick-level or irregular
    minute_close_data = df['close'].resample('min').last()
    
    # Reindex the minute-level 'close' data to the full minute index
    # This will introduce NaN for minutes where there's no original data after resampling
    daily_close_resampled = minute_close_data.reindex(full_minute_index).to_frame()
    
    # Define day and night session time ranges
    day_start_time = pd.to_datetime('08:45').time()
    day_end_time = pd.to_datetime('13:45').time() # Exclusive
    night_start_time = pd.to_datetime('15:00').time()
    night_end_time = pd.to_datetime('05:00').time() # Exclusive, next day

    time_only = daily_close_resampled.index.time

    # Create boolean masks for each session
    is_day_session = (time_only >= day_start_time) & \
                     (time_only < day_end_time)

    is_night_session = ((time_only >= night_start_time) & \
                        (time_only <= pd.to_datetime('23:59:59').time())) | \
                       (time_only < night_end_time)

    # Create session-specific series, setting non-session times to NaN
    # Plotly will automatically break lines at NaN values
    # Apply ffill before applying session masks to ensure continuity within sessions
    # Create session-specific series, setting non-session times to NaN
    # Plotly will automatically break lines at NaN values
    # Create session-specific series, setting non-session times to NaN
    # For continuity within sessions during flat price periods (e.g., limit up/down),
    # we ffill and bfill *within* the session boundaries.
    # This ensures lines are drawn for continuous flat prices without "filling" non-trading gaps.
    day_session_close = daily_close_resampled['close'].where(is_day_session)
    night_session_close = daily_close_resampled['close'].where(is_night_session)

    fig = go.Figure()

    # Add traces for day and night sessions only if they have non-NaN data
    if not day_session_close.dropna().empty:
        fig.add_trace(go.Scatter(x=day_session_close.index, y=day_session_close.values,
                                 mode='lines', name='日盤',
                                 line=dict(color='blue'),
                                 hovertemplate='%{fullData.name}<br>時間: %{x}<br>價格: %{y:,.0f}<extra></extra>',
                                 hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.7)", font_size=12, font_color="black")))

    if not night_session_close.dropna().empty:
        fig.add_trace(go.Scatter(x=night_session_close.index, y=night_session_close.values,
                                 mode='lines', name='夜盤',
                                 line=dict(color='red'),
                                 hovertemplate='%{fullData.name}<br>時間: %{x}<br>價格: %{y:,.0f}<extra></extra>',
                                 hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.7)", font_size=12, font_color="black")))

    # Add green rectangles for sessions with no data (holidays)
    shapes = []
    all_dates = pd.to_datetime(df.index.date).unique()

    for current_date in all_dates:
        current_date_ts = pd.Timestamp(current_date)

        # Check Day Session
        day_session_start_dt = current_date_ts.replace(hour=day_start_time.hour, minute=day_start_time.minute, second=day_start_time.second)
        day_session_end_dt = current_date_ts.replace(hour=day_end_time.hour, minute=day_end_time.minute, second=day_end_time.second)
        
        # Check if there's any actual data (not just ffilled NaNs) in the original df for this day session
        day_data_exists = df.index.to_series().between(day_session_start_dt, day_session_end_dt, inclusive='both').any()
        
        if not day_data_exists: # If no original data for this day session
            shapes.append(
                dict(
                    type="rect",
                    xref="x", yref="paper",
                    x0=day_session_start_dt, y0=0,
                    x1=day_session_end_dt, y1=1,
                    fillcolor="rgba(0,255,0,0.2)", # Green with transparency
                    layer="below",
                    line_width=0,
                )
            )

        # Check Night Session
        night_session_start_dt = current_date_ts.replace(hour=night_start_time.hour, minute=night_start_time.minute, second=night_start_time.second)
        night_session_end_dt = (current_date_ts + pd.Timedelta(days=1)).replace(hour=night_end_time.hour, minute=night_end_time.minute, second=night_end_time.second)

        # Check if there's any actual data in the original df for this night session
        night_data_exists_part1 = df.index.to_series().between(night_session_start_dt, current_date_ts.replace(hour=23, minute=59, second=59), inclusive='both').any()
        night_data_exists_part2 = df.index.to_series().between((current_date_ts + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0), night_session_end_dt, inclusive='both').any()
        night_data_exists = night_data_exists_part1 or night_data_exists_part2
        
        if not night_data_exists: # If no original data for this night session
            shapes.append(
                dict(
                    type="rect",
                    xref="x", yref="paper",
                    x0=night_session_start_dt, y0=0,
                    x1=night_session_end_dt, y1=1,
                    fillcolor="rgba(0,255,0,0.2)",
                    layer="below",
                    line_width=0,
                )
            )
    
    fig.update_layout(shapes=shapes)

    fig.update_layout(
        title_text='每日收盤價互動式走勢圖 (日盤/夜盤區分)',
        xaxis_rangeslider_visible=True,
        xaxis_title='時間',
        yaxis_title='價格',
        hovermode='closest', # 只顯示最接近滑鼠的數據點信息
        height=700, # 再次增加高度
        margin=dict(b=100), # 增加底部邊距
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            type="date"
        )
    )
    fig.show()

def plot_daily_volume(df: pd.DataFrame):
    """
    繪製每日總成交量變化圖。

    Args:
        df (pd.DataFrame): 包含 'volume' 欄位的 DataFrame，索引為時間。
    """
    plt.rcParams['font.family'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 6))
    daily_volume = df['volume'].resample('D').sum()
    plt.bar(daily_volume.index, daily_volume.values, label='每日總成交量')
    plt.title('每日總成交量變化圖')
    plt.xlabel('時間')
    plt.ylabel('成交量')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()


def analyze_daily_volatility(df: pd.DataFrame):
    """
    進行基礎波動率分析 (日報酬率標準差)。

    Args:
        df (pd.DataFrame): 包含 'close' 欄位的 DataFrame，索引為時間。
    """
    daily_returns = df['close'].resample('D').ffill().pct_change().dropna()
    daily_volatility = daily_returns.std()
    print(f"\n日報酬率標準差 (波動率): {daily_volatility:.4f}")

def plot_price_volume_distribution(df: pd.DataFrame):
    """
    繪製價格與成交量分佈圖。

    Args:
        df (pd.DataFrame): 包含 'close' 和 'volume' 欄位的 DataFrame。
    """
    plt.rcParams['font.family'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['close'], kde=False) # 關閉KDE以提高速度
    plt.title('價格分佈')
    plt.xlabel('價格')
    plt.ylabel('頻率')

    plt.subplot(1, 2, 2)
    sns.histplot(df['volume'], kde=False, bins=100)
    plt.title('成交量分佈')
    plt.xlabel('成交量')
    plt.ylabel('頻率')
    plt.yscale('log') # 將Y軸設置為對數刻度
    plt.tight_layout()
    plt.show()

def plot_price_volume_scatter(df: pd.DataFrame):
    """
    繪製價格與成交量的氣泡圖，氣泡大小表示該區域的數據點密度。

    Args:
        df (pd.DataFrame): 包含 'close' (價格) 和 'volume' (成交量) 欄位的 DataFrame。
    """
    plt.rcParams['font.family'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 設定分箱數量
    num_bins_price = 50
    num_bins_volume = 50

    # 計算價格和成交量的 2D 頻率分佈
    # range 參數定義了每個維度的範圍，確保所有數據點都被包含
    # bins 參數定義了每個維度的分箱數量
    hist, xedges, yedges = np.histogram2d(df['close'], df['volume'],
                                          bins=[num_bins_price, num_bins_volume],
                                          range=[[df['close'].min(), df['close'].max()],
                                                 [df['volume'].min(), df['volume'].max()]])

    # 創建一個 DataFrame 來儲存氣泡圖的數據
    # x 和 y 座標為 bin 的中心點，size 為頻率
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    # 將 2D 頻率分佈轉換為適合 seaborn.scatterplot 的長格式 DataFrame
    plot_data = []
    for i in range(len(x_centers)):
        for j in range(len(y_centers)):
            if hist[i, j] > 0: # 只包含有數據的 bin
                plot_data.append({
                    'price_center': x_centers[i],
                    'volume_center': y_centers[j],
                    'frequency': hist[i, j]
                })
    grouped_data = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=grouped_data,
        x='price_center',
        y='volume_center',
        size='frequency', # 氣泡大小由頻率決定
        sizes=(20, 2000), # 氣泡大小範圍，可以調整
        hue='frequency', # 顏色也可以由頻率決定，增加視覺效果
        palette='viridis', # 配色方案
        legend='full',
        alpha=0.7
    )
    plt.title('價格與成交量密度氣泡圖')
    plt.xlabel('價格')
    plt.ylabel('成交量')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()