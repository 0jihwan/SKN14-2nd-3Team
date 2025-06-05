import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm, rc

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rc

font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path)
rc('font', family=font_prop.get_name())

plt.rc('axes', unicode_minus=False) # matplotlib이 기본적으로 사용하는 유니코드 마이너스 비활성화, 아스키코드 마이너스 사용)

st.set_page_config(layout="wide")
plt.rc('font', family='Malgun Gothic')  # 한글 폰트

# 제목
st.header("📈 2. 고객 이탈 분석 대시보드")

# 데이터 로드
df = pd.read_csv('data/CommerceData.csv')

st.subheader("📋 원본 데이터")
st.dataframe(df.head())

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


#st.set_page_config(layout="wide")
st.title("📊 이탈 분석: 데이터 전처리")
st.subheader("🔧 데이터 전처리")

# Mobile Phone -> Mobile
df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Mobile Phone': 'Mobile'})

# 수치형 평균값 대체
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('CustomerID')
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# 범주형 최빈값 대체
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 불필요 컬럼 제거 및 데이터 타입 변경
df = df.drop(columns=['CustomerID'])
df['Churn'] = df['Churn'].astype('category')

st.success("✅ 전처리 완료")
st.dataframe(df.head())

st.subheader("📈 변수별 시각화")

# SatisfactionScore
st.markdown("**💡 Satisfaction Score vs Churn**")
fig1, ax1 = plt.subplots()
sns.boxplot(x='Churn', y='SatisfactionScore', data=df, ax=ax1)
ax1.set_title("만족도 점수별 이탈률")
st.pyplot(fig1)

# HourSpendOnApp
st.markdown("**💡 App Usage Time vs Churn**")
fig2, ax2 = plt.subplots()
sns.boxplot(x='Churn', y='HourSpendOnApp', data=df, ax=ax2)
ax2.set_title("앱 사용시간과 이탈률")
st.pyplot(fig2)

# Complain
st.markdown("**💡 Complain 비율 vs Churn**")
fig3, ax3 = plt.subplots()
sns.countplot(x='Complain', hue='Churn', data=df, ax=ax3)
ax3.set_title("불만여부별 이탈률")
st.pyplot(fig3)

# OrderCount
st.markdown("**💡 Order Count vs Churn**")
fig4, ax4 = plt.subplots()
sns.boxplot(x='Churn', y='OrderCount', data=df, ax=ax4)
ax4.set_title("주문 수와 이탈률")
st.pyplot(fig4)

# CouponUsed
st.markdown("**💡 Coupon Usage vs Churn**")
fig5, ax5 = plt.subplots()
sns.boxplot(x='Churn', y='CouponUsed', data=df, ax=ax5)
ax5.set_title("쿠폰 사용별 이탈률")
st.pyplot(fig5)


df = pd.read_csv('data/E_Commerce_Dataset_model2.csv')


# 이탈 현황 요약
total_customers = len(df)
churned = df['Churn'].sum()
retained = total_customers - churned
churn_rate = (churned / total_customers) * 100
retention_rate = (retained / total_customers) * 100

# 이탈 고객 파이차트
st.subheader("🧩 고객 이탈 비율")
fig1, ax1 = plt.subplots()
sizes = [retained, churned]
labels = [
    f'유지 고객\n({retained:,}명, {retention_rate:.1f}%)',
    f'이탈 고객\n({churned:,}명, {churn_rate:.1f}%)'
]
colors = ['#66b3ff', '#ff9999']
explode = (0.05, 0.05)

ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, shadow=True, textprops={'fontsize': 12})
ax1.set_title('고객 이탈 현황', pad=20, size=15)
st.pyplot(fig1)

# 대시보드 시각화
st.subheader("📊 이탈 관련 상세 분석")
fig2, axs = plt.subplots(2, 3, figsize=(20, 12))

# 1. 만족도 점수별 이탈률
churn_by_satisfaction = df.groupby('SatisfactionScore')['Churn'].mean() * 100
sns.barplot(x=churn_by_satisfaction.index, y=churn_by_satisfaction.values, ax=axs[0, 0], palette='RdYlBu_r')
axs[0, 0].set_title("만족도 점수별 이탈률")

# 2. 도시 등급별 이탈률
churn_by_city = df.groupby('CityTier')['Churn'].mean() * 100
sns.barplot(x=churn_by_city.index, y=churn_by_city.values, ax=axs[0, 1], palette='viridis')
axs[0, 1].set_title("도시 등급별 이탈률")

# 3. 앱 사용시간과 주문 수
sns.scatterplot(data=df, x='HourSpendOnApp', y='OrderCount',
                hue='Churn', palette=['blue', 'red'], alpha=0.6, ax=axs[0, 2])
axs[0, 2].set_title("앱 사용시간 vs 주문 수")

# 4. 불만 여부
churn_by_complain = df.groupby('Complain')['Churn'].mean() * 100
sns.barplot(x=['불만 없음', '불만 있음'], y=churn_by_complain.values, ax=axs[1, 0], palette='Set2')
axs[1, 0].set_title("불만 여부별 이탈률")

# 5. 주문 금액 증가율
sns.boxplot(data=df, x='Churn', y='OrderAmountHikeFromlastYear', ax=axs[1, 1], palette=['blue', 'red'])
axs[1, 1].set_title("이탈 여부 vs 주문 금액 증가율")

# 6. 등록 기기 수
churn_by_devices = df.groupby('NumberOfDeviceRegistered')['Churn'].mean() * 100
sns.barplot(x=churn_by_devices.index, y=churn_by_devices.values, ax=axs[1, 2], palette='mako')
axs[1, 2].set_title("기기 수별 이탈률")

plt.tight_layout()
st.pyplot(fig2)

# 통계 출력
st.subheader("📌 이탈 상세 통계")
st.write(f"전체 고객 수: **{total_customers:,}명**")
st.write(f"유지 고객 수: **{retained:,}명** ({retention_rate:.1f}%)")
st.write(f"이탈 고객 수: **{churned:,}명** ({churn_rate:.1f}%)")

df = pd.read_csv('data/E Commerce Dataset22.csv')

# -------------------
# 가설 1
# -------------------
st.header("가설 1️⃣: 만족도가 낮을수록 고객은 이탈할 가능성이 높다")

fig1, ax1 = plt.subplots()
sns.barplot(data=df, x='SatisfactionScore', y='Churn', ax=ax1)
ax1.set_title("만족도별 이탈률")
st.pyplot(fig1)

# t-test
group0 = df[df['Churn'] == 0]['SatisfactionScore']
group1 = df[df['Churn'] == 1]['SatisfactionScore']
t_stat, p_value = ttest_ind(group0, group1, equal_var=False)

st.subheader("📊 독립표본 t-검정 결과")
st.write('**t-statistic** :', round(t_stat, 4))
st.write('**p-value** :', round(p_value, 20)) #round(p_value, 4)

if p_value < 0.05:
    st.success("📌 p-value가 0.05보다 작으므로, 이탈 고객과 잔존 고객 간 만족도에 유의미한 차이가 있습니다.")
else:
    st.warning("p-value가 0.05보다 크므로, 만족도 차이가 통계적으로 유의하지 않습니다.")

# -------------------
# 가설 2
# -------------------
st.header("가설 2️⃣: 불만을 제기한 고객은 이탈할 가능성이 높다")

fig2, ax2 = plt.subplots()
sns.barplot(data=df, x='Complain', y='Churn', ax=ax2)
ax2.set_xticklabels(['불만 없음', '불만 있음'])
ax2.set_title("불만 제기 여부에 따른 이탈률")
st.pyplot(fig2)

# 카이제곱 검정
st.subheader("📊 카이제곱 독립성 검정 결과")
table = pd.crosstab(df['Complain'], df['Churn'])
chi2, p, dof, expected = chi2_contingency(table)

st.write('**Chi-squared 통계량** :', round(chi2,4))
st.write('**p-value** :', round(p,10))

if p < 0.05:
    st.success("📌 p-value가 0.05보다 작으므로, 불만 제기 여부와 이탈 간에는 통계적으로 유의미한 관계가 있습니다.")
else:
    st.warning("p-value가 0.05보다 크므로, 불만 제기 여부와 이탈 간의 관계는 유의하지 않습니다.")

# -------------------
# 가설 3
# -------------------
st.header("가설 3️⃣: 마지막 주문 이후 경과일이 클수록 이탈할 가능성이 높다")

fig3, ax3 = plt.subplots()
sns.boxplot(data=df, x='Churn', y='DaySinceLastOrder', ax=ax3)
ax3.set_xticklabels(['유지 고객', '이탈 고객'])
ax3.set_title("DaySinceLastOrder vs 이탈 여부")
st.pyplot(fig3)

# t-test
st.subheader("📊 독립표본 t-검정 결과")

df = df.dropna(subset=['Churn', 'DaySinceLastOrder'])

group0 = df[df['Churn'] == 0]['DaySinceLastOrder']
group1 = df[df['Churn'] == 1]['DaySinceLastOrder']

t_stat, p_value = ttest_ind(group0, group1, equal_var=False)


st.write('**t-statistic** :', round(t_stat, 4))
st.write('**p-value** :', round(p_value, 20))

if p_value < 0.05:
    st.success("📌 p-value가 0.05보다 작으므로, 마지막 주문일로부터 경과일수는 이탈 여부에 유의미한 영향을 미칩니다.")
else:
    st.warning("p-value가 0.05보다 크므로, 마지막 주문 경과일과 이탈 간 유의미한 차이는 없습니다.")


st.title("고객 만족도 그룹 분석 및 이탈률 통계")

st.subheader("1. 만족도 그룹 생성")

def satisfaction_group(score):
    if score <= 2:
        return 'Low (1-2)'
    elif score == 3:
        return 'Medium (3)'
    else:
        return 'High (4-5)'

df['SatisfactionGroup'] = df['SatisfactionScore'].apply(satisfaction_group)
st.dataframe(df[['SatisfactionScore', 'SatisfactionGroup']].head())

st.subheader("2. 만족도 그룹별 이탈자 수")

fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x='SatisfactionGroup', hue='Churn', ax=ax1)
ax1.set_title('만족도 그룹별 이탈자 수')
ax1.set_xlabel('만족도 그룹')
ax1.set_ylabel('고객 수')
ax1.legend(title='Churn (1=이탈)')
st.pyplot(fig1)

st.subheader("3. 만족도 그룹별 이탈 비율")

grouped = df.groupby(['SatisfactionGroup', 'Churn']).size().reset_index(name='count')
total_by_group = grouped.groupby('SatisfactionGroup')['count'].transform('sum')
grouped['ratio'] = grouped['count'] / total_by_group

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(data=grouped, x='SatisfactionGroup', y='ratio', hue='Churn', ax=ax2)
ax2.set_title('만족도 그룹별 이탈 비율')
ax2.set_xlabel('만족도 그룹')
ax2.set_ylabel('비율')
ax2.legend(title='Churn (1=이탈)')
st.pyplot(fig2)

st.subheader("4. 만족도 그룹과 이탈 여부의 교차표 (count)")

pivot_table = pd.pivot_table(
    df,
    index='SatisfactionGroup',
    columns='Churn',
    values='Gender',  # 아무 변수로 count 계산
    aggfunc='count',
    fill_value=0,
    margins=True,
    margins_name='총합'
)
st.dataframe(pivot_table)

st.subheader("5. 카이제곱 검정을 통한 통계 분석")

contingency = pd.crosstab(df['SatisfactionGroup'], df['Churn'])
chi2, p, dof, expected = chi2_contingency(contingency)

st.write('Chi2 통계량:', round(chi2,4))
st.write('p-value:', round(p,6))
st.write(f"자유도: {dof}")
st.write("기대값 테이블:")
st.dataframe(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns))

st.title("ERD 다이어그램")
from PIL import Image
image = Image.open('data/ERDdiagram.png')
st.image(image)

