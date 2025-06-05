import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm, rc

font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path)
rc('font', family=font_prop.get_name())

plt.rc('axes', unicode_minus=False) # matplotlib이 기본적으로 사용하는 유니코드 마이너스 비활성화, 아스키코드 마이너스 사용)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils.preprocessing import preprocess_commerce_data, split_and_scale, preprocess_all_data, predict_churn1
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 모델 경로 및 threshold 설정
model_options = {
    "XGBoost": ("../03_trained_model/xgboost_model(threshold=0.075).pkl", 0.075),
    "GradientBoosting": ("../03_trained_model/gb_model(threshold=0.1375).pkl", 0.1375),
    "RandomForest": ("../03_trained_model/random_forest_model(threshold=0.225).pkl", 0.225),
    "HistGradientBoosting": ("../03_trained_model/hgb_model(threshold=0.0001).pkl", 0.0001),
    "Stacking": ("../03_trained_model/stack_model(threshold=0.091).pkl", 0.091),
    "Voting": ("../03_trained_model/voting_model(threshold=0.1925).pkl", 0.137),
    "LGBM": ("../03_trained_model/lgbm(threshold=0.147).pkl", 0.147)
}

st.header("3. 모델 학습 및 평가")

# 평가 방식 선택
view_mode = st.radio("🧪 평가 방식 선택", ["개별 모델 상세 보기", "모든 모델 비교", "클러스터링"], horizontal=True)

# 데이터 불러오기 및 전처리
df_raw = pd.read_csv('data/CommerceData.csv')
df_processed = preprocess_commerce_data(df_raw)

exclude = ['CityTier', 'PreferredPaymentMode', 'Gender',
           'PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice']

X_train, X_test, y_train, y_test, scaler = split_and_scale(
    df_processed, target_col='Churn', exclude_cols=exclude
)

if view_mode == "개별 모델 상세 보기":
    selected_model = st.selectbox("🔍 확인할 모델을 선택하세요", list(model_options.keys()))
    model_path, threshold = model_options[selected_model]
    model = joblib.load(model_path)

    predict_mode = st.radio("예측 방식 선택", ["Threshold 적용", "기본 예측"], horizontal=True)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None  # 혹시나 대비

    if predict_mode == "Threshold 적용" and y_proba is not None:
        y_pred = (y_proba >= threshold).astype(int)
        st.info(f"📌 Threshold `{threshold}` 기준으로 분류되었습니다.")
    else:
        y_pred = model.predict(X_test)
        st.info("📌 기본 예측 결과입니다.")

    y_true = y_test

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    st.markdown(f"""
    - 사용 모델: **{selected_model}**
    - 평가 기준(Test셋 데이터): Accuracy, Recall, Precision, F1 Score
    
    📌 **성능 결과**
    - Accuracy: `{acc:.4f}`  
    - Recall: `{rec:.4f}`  
    - Precision: `{prec:.4f}`  
    - F1 Score: `{f1:.4f}`
    """)

    metrics = {"Accuracy": acc, "Recall": rec, "Precision": prec, "F1 Score": f1}
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values(), color=["skyblue", "salmon", "lightgreen", "violet"])
    ax.set_ylim(0, 1)
    ax.set_title(f"{selected_model} 성능 지표 - {predict_mode}")
    st.pyplot(fig)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"{selected_model} ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📋 Classification Report")
    accuracy_val = report.pop("accuracy")
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df)
    st.markdown(f"✅ **Overall Accuracy**: `{accuracy_val:.4f}`")

elif view_mode == "모든 모델 비교":
    st.subheader("모델 성능 비교 (기본 vs Threshold)")

    compare_results = []

    for model_name, (path, threshold) in model_options.items():
        model = joblib.load(path)

        # Default predict()
        y_pred_default = model.predict(X_test)

        # Threshold predict_proba()
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred_thresh = (y_proba >= threshold).astype(int)
        else:
            y_pred_thresh = y_pred_default

        y_true = y_test

        for version, y_pred in zip(["Default", "Threshold"], [y_pred_default, y_pred_thresh]):
            report = classification_report(y_true, y_pred, output_dict=True)
            accuracy = report["accuracy"]
            pos_recall = report["1"]["recall"]

            compare_results.append({
                "Model": model_name,
                "Version": version,
                "Accuracy": accuracy,
                "Positive Recall": pos_recall
            })

    # 결과 프레임 변환
    compare_df = pd.DataFrame(compare_results)

    # 최고 모델 찾기
    best_row = compare_df[compare_df["Positive Recall"] == 1.0].sort_values(
        "Accuracy", ascending=False
    ).head(1)
    best_model_name = best_row["Model"].values[0]
    best_version = best_row["Version"].values[0]

    # "Best" 컬럼 추가
    compare_df["Best"] = compare_df.apply(
        lambda row: "⭐ 최고 모델" if (row["Model"] == best_model_name and row["Version"] == best_version) else "", axis=1
    )

    # 정렬 (Positive Recall 기준 → Accuracy 기준)
    compare_df = compare_df.sort_values(
        by=["Positive Recall", "Accuracy"], ascending=[False, False]
    ).reset_index(drop=True)

    # 표시
    st.dataframe(compare_df.round(4))

    # Positive Recall 시각화 추가
    fig, ax = plt.subplots(figsize=(10, 5))  # 그래프도 살짝 넓혀줌
    sns.barplot(data=compare_df, x="Model", y="Positive Recall", hue="Version", ax=ax)
    ax.set_title("모델별 Positive Recall 비교")
    ax.set_ylim(0.85, 1.01)  # 또는 0.9 ~ 1.0 구간
    plt.xticks(rotation=30, ha='right')  # ✅ 라벨 회전
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))  # 그래프도 살짝 넓혀줌
    sns.barplot(data=compare_df, x="Model", y="Accuracy", hue="Version", ax=ax, palette='Set2')
    ax.set_title("모델별 Accuracy 비교")
    ax.set_ylim(0.85, 1.01)  # 또는 0.9 ~ 1.0 구간
    plt.xticks(rotation=30, ha='right')  # ✅ 라벨 회전
    st.pyplot(fig)

elif view_mode == "클러스터링":
    st.subheader("📊 고객 군집 분석 (KMeans)")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    # RFM 데이터 준비
    df_processed1 = preprocess_all_data(df_raw)
    rfm_df = df_processed1[['DaySinceLastOrder', 'OrderCount', 'CashbackAmount']].copy()
    rfm_df.columns = ['recency', 'frequency', 'monetary']

    scaler_rfm = joblib.load("../03_trained_model/kmeans_scaler.pkl")
    rfm_scaled = scaler_rfm.transform(rfm_df)

    # inertia & silhouette 계산
    inertias = []
    silhouettes = []
    k_range = range(2, 8)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(rfm_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(rfm_scaled, labels))

    # 그래프 출력
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(k_range, inertias, marker='o', label='Inertia')
    ax1.set_ylabel('Inertia')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_title('Elbow & Silhouette Score by k')

    ax2 = ax1.twinx()
    ax2.plot(k_range, silhouettes, marker='s', color='green', label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    st.pyplot(fig)

    # k=6 기준 클러스터링
    st.markdown("### ✅ 최적 군집 수: k = 6")
    kmeans = joblib.load('../03_trained_model/kmeans.pkl')
    cluster_labels = kmeans.predict(rfm_scaled)
    rfm_df['cluster'] = cluster_labels

    # 예측 확률 붙이기 (모델 불러오기)
    from sklearn.ensemble import GradientBoostingClassifier

    df_labeled, df_scaled, df_result = predict_churn1(df_raw)
    rfm_df['model_pred_proba'] = df_result['Churn_Prob']

    # 클러스터별 평균 요약
    cluster_summary = rfm_df.groupby('cluster')[
        ['recency', 'frequency', 'monetary', 'model_pred_proba']].mean().round(4).reset_index()

    # 특성 및 전략 설명 추가
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
    # 클러스터 번호 기준 정렬 맞추기 위해 인덱스 순서로 매핑
    cluster_summary['고객 특성'] = [descriptions[i][0] for i in cluster_summary['cluster']]
    cluster_summary['추천 전략'] = [descriptions[i][1] for i in cluster_summary['cluster']]

    # 클러스터 요약 표 출력 (RFM + 예측 확률만)
    st.markdown("#### 📋 클러스터별 RFM + 이탈 확률 요약")
    rfm_summary_only = cluster_summary[['cluster', 'recency', 'frequency', 'monetary', 'model_pred_proba']]
    st.dataframe(rfm_summary_only)

    # 클러스터별 특성과 전략 출력
    st.markdown("#### 📌 클러스터별 고객 특성 및 추천 전략")
    for i, row in cluster_summary.iterrows():
        st.markdown(f"""
        🔹 **클러스터 {row['cluster']}**
        - **고객 특성**: {row['고객 특성']}
        - **추천 전략**: {row['추천 전략']}
        ---
        """)
