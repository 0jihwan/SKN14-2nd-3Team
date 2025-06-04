import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_predict_data():
    np.random.seed(42)
    return pd.DataFrame({
        'CustomerID': range(1, 101),
        'Churn_Prob': np.round(np.random.uniform(0.1, 0.95, size=100), 2),
        'Churn': np.random.choice([0, 1], size=100),
        'Cluster': np.random.choice(['A', 'B', 'C'], size=100)
    })

st.header("4. 이탈 예측 및 클러스터링 결과")
df = load_predict_data()
st.dataframe(df)

st.subheader("📊 클러스터별 이탈 확률 분포")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='Cluster', y='Churn_Prob')
st.pyplot(fig)