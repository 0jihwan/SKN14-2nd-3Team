import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_commerce_data(df):
    df = df.copy()

    # 1. 결측치 처리
    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
    df['WarehouseToHome'] = df['WarehouseToHome'].fillna(df['WarehouseToHome'].median())
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].median())

    df['NoLastYearPurchase'] = df['OrderAmountHikeFromlastYear'].isna().astype(int)
    df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(0)

    df['CouponUsed'] = df['CouponUsed'].fillna(0)
    df['OrderCount'] = df['OrderCount'].fillna(0)

    # 주문 안 한 고객 처리
    max_day = df['DaySinceLastOrder'].max()
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(max_day + 1)
    df['NeverOrdered'] = (df['DaySinceLastOrder'] > max_day).astype(int)

    # 2. 범주형 인코딩
    cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                'PreferedOrderCat', 'MaritalStatus']

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  # 나중에 inverse_transform 등을 위해 저장

    # 3. ID 제거
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)

    return df, le_dict



def split_and_scale(df, target_col='Churn', exclude_cols=None, test_size=0.2, random_state=42):
    """
    연속형 변수만 StandardScaler로 스케일링하여 train/test로 나누는 함수.

    Parameters:
    - df (pd.DataFrame): 입력 데이터
    - target_col (str): 예측할 타겟 컬럼
    - exclude_cols (list): 스케일링에서 제외할 컬럼들
    - test_size (float): 테스트셋 비율
    - random_state (int): 랜덤 시드

    Returns:
    - X_train, X_test, y_train, y_test, scaler
    """

    if exclude_cols is None:
        exclude_cols = []

    # 1. Feature, Target 분리
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Train/Test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. 연속형 변수만 선택
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols)

    # 4. 스케일링
    scaler = StandardScaler()
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler
