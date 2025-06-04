
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data

import matplotlib
import matplotlib.font_manager as fm
font_path = "C:\\Windows\\Fonts\\H2GTRM.TTF" # 윈도우OS 폰트경로
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name() # 폰트명
matplotlib.rc('font', family=font_name)
plt.rc('axes', unicode_minus=False) # matplotlib이 기본적으로 사용하는 유니코드 마이너스 비활성화, 아스키코드 마이너스 사용)

st.header("2. 데이터 개요")

df = load_data()
if df is not None:
    st.subheader('데이터셋 정보')
    st.write(f"🧾 데이터 행 개수(데이터 샘플 수): {df.shape[0]}, 열 개수(데이터 특성 수): {df.shape[1]}")

    st.subheader("📋 데이터셋 정보(info) 요약표")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Null Count": df.isnull().sum(),
        "Unique Count": df.nunique(),
    }).reset_index(drop=True).set_index('Column')

    st.dataframe(info_df)

    st.subheader("📊 df.describe(include='all') 통계 요약")

    desc = df.describe(include='all').transpose().reset_index()
    desc.rename(columns={'index': 'Column'}, inplace=True)

    st.dataframe(desc)

    exclude_cols = ['CustomerID']

    # 명시적으로 지정
    categorical_cols = [
        'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
        'PreferedOrderCat', 'MaritalStatus', 'CityTier', 'SatisfactionScore'
    ]
    numeric_cols = [
        col for col in df.columns
        if col not in categorical_cols + exclude_cols + ['Churn']
    ]

    # Complain은 수치형으로 포함
    if 'Complain' in df.columns:
        numeric_cols.append('Complain')

    st.title("📊 변수별 분포 시각화")

    # 1. 🎯 타겟 변수 Churn 분포
    st.subheader("🎯 Churn 분포")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Churn', ax=ax1, palette='Set2')
    ax1.set_title("Churn (이탈 여부) 분포")
    st.pyplot(fig1)

    # 2. 📈 수치형 변수 분포 (Histogram)
    st.subheader("📈 수치형 변수 분포")
    selected_num = st.multiselect("시각화할 수치형 변수 선택", numeric_cols, default=numeric_cols[:3])

    for col in selected_num:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color='skyblue')
        ax.set_title(f"{col} Distribution")
        st.pyplot(fig)

    # 3. 📊 범주형 변수 분포 (Count Plot)
    st.subheader("📊 범주형 변수 분포")
    selected_cat = st.multiselect("시각화할 범주형 변수 선택", categorical_cols, default=categorical_cols[:3])

    for col in selected_cat:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax, palette='pastel')
        ax.set_title(f"{col} Value Counts")
        plt.xticks(rotation=30)
        st.pyplot(fig)