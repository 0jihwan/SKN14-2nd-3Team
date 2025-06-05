import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

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
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median())

    # 2. 범주형 인코딩

    cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                'PreferedOrderCat', 'MaritalStatus']

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 3. ID 제거
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)

    return df



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


def preprocess_for_prediction(df):

    df = df.copy()

    # 1. 결축치 처리
    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
    df['WarehouseToHome'] = df['WarehouseToHome'].fillna(df['WarehouseToHome'].median())
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].median())

    df['NoLastYearPurchase'] = df['OrderAmountHikeFromlastYear'].isna().astype(int)
    df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(0)

    df['CouponUsed'] = df['CouponUsed'].fillna(0)
    df['OrderCount'] = df['OrderCount'].fillna(0)

    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median())

    # 2. 범주형 인코딩
    cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                'PreferedOrderCat', 'MaritalStatus']

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df.drop(columns=['CustomerID'], inplace=True)

    return df



def predict_churn(df_new):
    # 전처리
    df_pred_input = preprocess_for_prediction(df_new)

    # 스케일링용 변수 및 스케일러 로드
    scaler = joblib.load("../03_trained_model/train_scaler.pkl")

    exclude = ['CityTier', 'PreferredPaymentMode', 'Gender',
               'PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice']

    # 정수형, 실수형 컬럼 중에서 exclude 컬럼 제외한 컬럼 뽑기
    num_cols = df_pred_input.select_dtypes(include=['float64', 'int64']).columns.difference(exclude)

    df_pred = df_pred_input.copy()

    df_pred[num_cols] = scaler.transform(df_pred[num_cols])


    # 모델 불러오기 및 예측
    model = joblib.load("../03_trained_model/gb_model(threshold=0.1375).pkl")
    threshold = 0.1375
    y_proba = model.predict_proba(df_pred)[:, 1]
    y_pred = (y_proba > threshold).astype(int)

    # 예측 결과 원본 데이터에 붙이기
    df_result = df_new.copy()
    df_result['Churn_Prob'] = y_proba
    df_result['Churn_Pred'] = y_pred

    return df_pred_input, df_pred, df_result

def preprocess_for_kmeans(df):

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
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median())

    cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                'PreferedOrderCat', 'MaritalStatus']

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df.drop(columns=['CustomerID', 'Churn'], inplace=True)

    return df

def preprocess_all_data(df):
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
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median())

    # 2. 범주형 인코딩

    cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                'PreferedOrderCat', 'MaritalStatus']

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 3. ID 제거
    df.drop(columns=['CustomerID', 'Churn'], inplace=True)

    return df

def predict_churn1(df_new):
    # 전처리
    df_pred_input = preprocess_for_kmeans(df_new)

    # 스케일링용 변수 및 스케일러 로드
    scaler = joblib.load("../03_trained_model/train_scaler.pkl")

    exclude = ['CityTier', 'PreferredPaymentMode', 'Gender',
               'PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice']

    # 정수형, 실수형 컬럼 중에서 exclude 컬럼 제외한 컬럼 뽑기
    num_cols = df_pred_input.select_dtypes(include=['float64', 'int64']).columns.difference(exclude)

    df_pred = df_pred_input.copy()

    df_pred[num_cols] = scaler.transform(df_pred[num_cols])


    # 모델 불러오기 및 예측
    model = joblib.load("../03_trained_model/gb_model(threshold=0.1375).pkl")
    y_proba = model.predict_proba(df_pred)[:, 1]

    # 예측 결과 원본 데이터에 붙이기
    df_result = df_new.copy()
    df_result['Churn_Prob'] = y_proba

    return df_pred_input, df_pred, df_result