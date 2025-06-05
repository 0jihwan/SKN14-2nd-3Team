import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/CommerceData.csv')
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return None