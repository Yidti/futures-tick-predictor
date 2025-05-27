import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, features: list, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    將 DataFrame 切分為訓練集和測試集。

    Args:
        df (pd.DataFrame): 包含特徵和目標的 DataFrame。
        features (list): 特徵欄位名稱列表。
        target (str): 目標欄位名稱。
        test_size (float): 測試集佔比。
        random_state (int): 隨機種子，用於重現性。

    Returns:
        tuple: 包含 X_train, X_test, y_train, y_test 的元組。
    """
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # 範例使用
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    features = ['feature1', 'feature2']
    target = 'target'

    X_train, X_test, y_train, y_test = split_data(df, features, target)

    print("X_train 形態:", X_train.shape)
    print("X_test 形態:", X_test.shape)
    print("y_train 形態:", y_train.shape)
    print("y_test 形態:", y_test.shape)

    print("\nX_train:\n", X_train)
    print("\ny_train:\n", y_train)