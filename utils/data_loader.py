import pandas as pd

def load_and_inspect_data(data_path):
    """
    載入歷史 Tick/分K 資料並進行初步檢視。
    """
    try:
        df = pd.read_parquet(data_path)
        print("資料載入成功！")
        print("前 5 行資料：")
        print(df.head())
        print("\n資料基本資訊：")
        df.info()
        print("\n描述性統計：")
        print(df.describe())
        return df
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {data_path}")
        return None
    except Exception as e:
        print(f"載入資料時發生錯誤：{e}")
        return None

def load_processed_data(file_path):
    """
    載入已處理的 Parquet 資料檔案。

    Args:
        file_path (str): Parquet 檔案的路徑。

    Returns:
        pd.DataFrame: 載入的 DataFrame，如果失敗則返回 None。
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"成功從 {file_path} 載入已處理的資料。")
        return df
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
        return None
    except Exception as e:
        print(f"載入檔案 {file_path} 時發生錯誤：{e}")
        return None
if __name__ == "__main__":
    # 指定資料檔案路徑
    data_path = './data/ticks_2025-04.parquet'
    
    # 載入資料並檢視
    df = load_and_inspect_data(data_path)
    
    if df is not None:
        print("\n資料欄位：")
        print(df.columns)
        print(f"資料筆數：{len(df)}")