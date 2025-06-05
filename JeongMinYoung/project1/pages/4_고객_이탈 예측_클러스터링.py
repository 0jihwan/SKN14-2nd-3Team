import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from utils.preprocessing import predict_churn
from utils.preprocessing import preprocess_for_prediction
from dotenv import load_dotenv
# Gemini API용
import os
import google.generativeai as genai

# 📌 Gemini API 키 불러오기
load_dotenv()
genai_api_key = os.environ.get('API_KEY')
# genai_api_key = "AIzaSyClJ8szfqSGJekr5bw9AxtsUhCFAjx9ruk"
if genai_api_key:
    genai.configure(api_key=genai_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 1. 모델 및 스케일러 로딩
rfm_scaler = joblib.load('../models/kmeans_scaler.pkl')
kmeans = joblib.load('../models/kmeans.pkl')

# 2. Streamlit 인터페이스
st.header("4. 고객 이탈 예측 및 맞춤형 마케팅 전략 도출")

uploaded = st.file_uploader("고객 데이터를 CSV 파일로 업로드하세요", type='csv')
if uploaded:
    df_new = pd.read_csv(uploaded)
    st.write("📋 업로드된 데이터:")
    st.dataframe(df_new.head(10))

    # 3. 이탈 예측 수행
    df_labeled, df_scaled, df_result = predict_churn(df_new)

    # 4. RFM 클러스터링
    rfm_df = df_labeled[['DaySinceLastOrder', 'OrderCount', 'CashbackAmount']].copy()
    rfm_df.columns = ['recency', 'frequency', 'monetary']
    rfm_df['CustomerID'] = df_new['CustomerID']
    rfm_scaled = rfm_scaler.transform(rfm_df[['recency', 'frequency', 'monetary']])
    clusters = kmeans.predict(rfm_scaled)

    rfm_df['cluster'] = clusters
    rfm_df['churn proba'] = df_result['Churn_Prob']
    rfm_df['churn pred'] = df_result['Churn_Pred']
    rfm_df = rfm_df[['CustomerID', 'recency', 'frequency', 'monetary', 'cluster', 'churn proba', 'churn pred']]

    # 결과 출력
    st.subheader("📈 예측 결과 및 클러스터")
    st.dataframe(rfm_df)

    st.subheader("👇 클러스터별 특성 요약 👇")
    summary_df = rfm_df.groupby('cluster')[['recency','frequency','monetary','churn proba']].mean().reset_index()
    st.dataframe(summary_df)

    st.subheader("📊 클러스터별 이탈 확률 분포")
    fig, ax = plt.subplots()
    sns.barplot(data=rfm_df, x='cluster', y='churn proba')
    st.pyplot(fig)

    # 클러스터별 고객 수 계산
    cluster_counts = rfm_df['cluster'].value_counts().sort_index().reset_index()
    cluster_counts.columns = ['cluster', '고객 수']

    st.subheader("✅ 클러스터별 고객 수")
    st.dataframe(cluster_counts)

    # 📌 Gemini API로 전략 생성
    if genai_api_key:
        st.subheader("🧠 Gemini 기반 클러스터 전략 자동 생성")
        if st.button("🤖 전략 자동 생성 요청"):
            with st.spinner("Gemini에게 전략 요청 중..."):
                result = []

                for _, row in summary_df.iterrows():
                    prompt = f"""
    다음은 클러스터 {int(row['cluster'])}의 평균 고객 특성입니다:

    - 📆 Recency (최근 구매일): {row['recency']:.2f} → 낮을수록 최근에 방문함
    - 🔁 Frequency (구매 빈도): {row['frequency']:.2f} → 높을수록 자주 구매함
    - 💰 Monetary (구매 금액): {row['monetary']:.2f} → 클수록 지출이 많음
    - ⚠️ Churn Probability (이탈 확률): {row['churn proba']:.4f} → 클수록 이탈 위험이 높음

    이 클러스터의 특성을 바탕으로 아래 조건을 고려하여 마케팅 전략을 한두 줄로 구체적으로 제안해주세요:

    - 이탈 가능성이 높은 고객 → 리텐션 중심 전략
    - 충성 고객 → 유지 및 프리미엄 혜택 중심 전략
    - 중간 고객 → 활성화, 관계 강화 전략

    전략은 마케터가 바로 사용할 수 있도록 간결하고 직관적으로 작성해주세요. 그리고 존댓말로 제안해주세요.
    """

                    try:
                        response = gemini_model.generate_content(prompt)
                        result.append((int(row['cluster']), response.text.strip()))
                    except Exception as e:
                        result.append((int(row['cluster']), f"⚠️ 오류: {e}"))

                for cluster_id, strategy in result:
                    st.markdown(f"### 🔹 클러스터 {cluster_id}")
                    st.success(strategy)

    # 📥 클러스터별 CSV 다운로드
    st.markdown("### 📥 클러스터별 고객 리스트 다운로드")
    for c in sorted(rfm_df['cluster'].unique()):
        cluster_data = rfm_df[rfm_df['cluster'] == c]
        csv = cluster_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=f"📁 클러스터 {c} 고객 리스트 다운로드",
            data=csv,
            file_name=f'cluster_{c}_customers.csv',
            mime='text/csv'
        )

    # 🔍 이탈 가능성이 높은 고객 (예측 결과 기준) 클러스터별 출력
    st.subheader("🚨 이탈 위험 고객 클러스터별 상세 보기")
    st.markdown("##### 아래는 **모델이 이탈할 것이라고 예측한 고객 리스트**입니다. \n(모델 이탈 예측 정확도: 테스트셋 기준 **99.37%**)")

    high_risk_df = rfm_df[rfm_df['churn pred'] == 1]

    if not high_risk_df.empty:
        st.warning(f"현재 총 **{len(high_risk_df)}명**의 고객이 이탈할 가능성이 높은 것으로 예측되었습니다.\n "
                   f"\n 각 클러스터별로 고객 특성에 맞는 **즉각적인 마케팅 전략** 도입이 필요합니다.\n "
                   f"\n 위에서 제공된 클러스터별 자동 전략 제안(Gemini 기반) 을 활용하여, 빠르게 대응 전략을 수립해보세요.")

        for cluster_id in sorted(high_risk_df['cluster'].unique()):
            cluster_subset = high_risk_df[high_risk_df['cluster'] == cluster_id]
            st.markdown(f"### 🔹 클러스터 {cluster_id} – 이탈 위험 고객 **{len(cluster_subset)}명**")
            st.dataframe(cluster_subset[['CustomerID', 'recency', 'frequency', 'monetary', 'churn proba']])

            # CSV 다운로드
            csv = cluster_subset.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label=f"📁 클러스터 {cluster_id} 이탈 위험 고객 리스트 다운로드",
                data=csv,
                file_name=f'high_risk_cluster_{cluster_id}.csv',
                mime='text/csv'
            )
    else:
        st.success("현재 이탈 위험 고객은 예측되지 않았습니다. 😊")