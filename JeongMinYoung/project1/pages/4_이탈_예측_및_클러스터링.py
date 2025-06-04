import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from utils.preprocessing import predict_churn

from utils.preprocessing import preprocess_for_prediction

# 1. 모델 및 스케일러 로딩
rfm_scaler = joblib.load('../models/kmeans_scaler.pkl')  # RFM용 StandardScaler
kmeans = joblib.load('../models/kmeans.pkl')             # k=6으로 학습된 KMeans

# 2. Streamlit 인터페이스
st.header("4. 신규 고객 데이터 업로드 및 군집 예측")

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

    # 결과 출력
    st.subheader("📈 예측 결과 및 클러스터")
    st.dataframe(rfm_df)

    st.subheader("👇 클러스터별 특성 요약 👇")
    st.dataframe(rfm_df.groupby('cluster')[['recency','frequency','monetary','churn proba']].mean())

    # 시각화
    st.subheader("📊 클러스터별 이탈 확률 분포")
    fig, ax = plt.subplots()
    sns.barplot(data=rfm_df, x='cluster', y='churn proba')
    st.pyplot(fig)

    # 클러스터별 고객 수 계산
    cluster_counts = rfm_df['cluster'].value_counts().sort_index().reset_index()
    cluster_counts.columns = ['cluster', '고객 수']

    # 전략 설명 매핑
    descriptions = [
        ("최근 방문했지만 구매 적음, 이탈 위험 매우 높음 (이탈률: 0.2215)",
         "즉각 리텐션 마케팅, 할인 알림 푸시"),  # cluster 0

        ("오래된 고객, 평균 이하 소비, 관계 단절 위험 (이탈률: 0.0912)",
         "재방문 유도, 리마인드 메시지 발송"),  # cluster 1

        ("최근 방문 + 고가 소비, 단발성 가능성 (이탈률: 0.1070)",
         "VIP 혜택 제안, 단기 고가 상품 추천"),  # cluster 2

        ("자주 구매 + 고지출, 핵심 로열 고객 (이탈률: 0.1199)",
         "후기 요청, 멤버십 제공, 로열티 강화"),  # cluster 3

        ("오래됐지만 고지출 유지, 충성 고객 (이탈률: 0.0632)",
         "프리미엄 혜택 리마인드, 감사 메시지"),  # cluster 4

        ("중간 활동, 중간 소비, 이탈 주의 고객 (이탈률: 0.1683)",
         "이탈 방지 쿠폰, 개인화 콘텐츠 제공")  # cluster 5
    ]

    desc_df = pd.DataFrame(descriptions, columns=['고객 특성', '추천 전략'])
    desc_df['cluster'] = desc_df.index

    # 병합해서 하나의 표로 정리
    cluster_summary = pd.merge(cluster_counts, desc_df, on='cluster')

    # 출력
    st.subheader("📌 클러스터별 고객 수 및 전략")
    st.dataframe(cluster_summary)

    for c in sorted(rfm_df['cluster'].unique()):
        cluster_data = rfm_df[rfm_df['cluster'] == c]
        csv = cluster_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=f"📥 클러스터 {c} 고객 CSV 다운로드",
            data=csv,
            file_name=f'cluster_{c}_customers.csv',
            mime='text/csv'
        )