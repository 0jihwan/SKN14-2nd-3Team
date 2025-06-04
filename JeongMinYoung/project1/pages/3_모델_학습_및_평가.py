import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils.preprocessing import preprocess_commerce_data, split_and_scale

# 모델 경로 및 threshold 설정
model_options = {
    "XGBoost": ("models/xgboost_model(threshold=0.21).pkl", 0.21),
    "GradientBoosting": ("models/gb_model(threshold=0.263).pkl", 0.263),
    "RandomForest": ("models/random_forest_model(threshold=0.285).pkl", 0.285)
}

st.header("3. 모델 학습 및 평가")

# 평가 방식 선택
view_mode = st.radio("🧪 평가 방식 선택", ["개별 모델 상세 보기", "모든 모델 비교"], horizontal=True)

# 데이터 불러오기 및 전처리
df_raw = pd.read_csv('data/CommerceData.csv')
df_processed, _ = preprocess_commerce_data(df_raw)

exclude = ['NeverOrdered', 'CityTier', 'PreferredPaymentMode', 'Gender',
           'PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice']

X_train, X_test, y_train, y_test, scaler = split_and_scale(
    df_processed, target_col='Churn', exclude_cols=exclude
)

if view_mode == "개별 모델 상세 보기":
    selected_model = st.selectbox("🔍 확인할 모델을 선택하세요", list(model_options.keys()))
    model_path, threshold = model_options[selected_model]
    model = joblib.load(model_path)

    predict_mode = st.radio("예측 방식 선택", ["Threshold 적용", "기본 predict()"], horizontal=True)

    if predict_mode == "Threshold 적용" and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        st.info(f"📌 Threshold `{threshold}` 기준으로 분류되었습니다.")
    else:
        y_pred = model.predict(X_test)
        st.info("📌 기본 `predict()` 결과입니다.")

    y_true = y_test

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    st.markdown(f"""
    - 사용 모델: **{selected_model}**
    - 평가 기준: Accuracy, Recall, Precision, F1 Score  
    📌 **성능 결과**
    - Accuracy: `{acc:.6f}`  
    - Recall: `{rec:.6f}`  
    - Precision: `{prec:.6f}`  
    - F1 Score: `{f1:.6f}`
    """)

    metrics = {"Accuracy": acc, "Recall": rec, "Precision": prec, "F1 Score": f1}
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values(), color=["skyblue", "salmon", "lightgreen", "violet"])
    ax.set_ylim(0, 1)
    ax.set_title(f"{selected_model} 성능 지표 - {predict_mode}")
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
            pos_recall = report["1"]["recall"] if "1" in report else 0.0

            compare_results.append({
                "Model": model_name,
                "Version": version,
                "Accuracy": accuracy,
                "Positive Recall": pos_recall
            })

    # DataFrame 변환
    compare_df = pd.DataFrame(compare_results)
    st.dataframe(compare_df.round(4))