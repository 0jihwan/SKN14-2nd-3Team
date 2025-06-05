import streamlit as st

# 페이지 설정
st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("💼 고객 이탈 예측 플랫폼")
st.markdown("고객 행동 데이터를 기반으로, 이탈 가능성을 예측하고 **클러스터별 맞춤 전략**까지 제시하는 통합 분석 플랫폼입니다.")

st.markdown("---")

# 소개 메시지
st.markdown("""
🎯 이 플랫폼에서는 아래 기능들을 확인할 수 있습니다:

1. 📖 **서비스 소개** – 이 프로젝트가 어떤 목적과 구조로 이루어졌는지 소개  
2. 🧾 **데이터 개요** – 고객 정보와 주요 변수에 대한 탐색  
3. 📊 **모델 학습 및 평가** – 다양한 모델을 통한 이탈 예측 성능 비교  
4. 🧠 **이탈 예측 및 클러스터링** – 고객 세분화 + 이탈 확률 예측 + 마케팅전략 제안 + 이탈 고위험 고객 추출

""")

st.markdown("👇 아래 버튼을 클릭하면 각 기능 페이지로 바로 이동할 수 있어요.")

st.markdown("---")

# 버튼 배치
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 서비스 소개 바로가기"):
        st.switch_page("pages/1_소개.py")

    if st.button("📊 모델 학습 및 평가"):
        st.switch_page("pages/3_모델_학습_및_평가.py")

with col2:
    if st.button("📁 데이터 개요 보기"):
        st.switch_page("pages/2_데이터_개요_분석내용.py")

    if st.button("🧠 이탈 예측 및 클러스터링 + 마케팅 전략 제안"):
        st.switch_page("pages/4_고객_이탈 예측_클러스터링.py")

st.markdown("---")
st.success("좌측 사이드바에서도 페이지를 자유롭게 이동할 수 있어요!")
