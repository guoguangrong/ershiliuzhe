# 导入核心库
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="AIS患者血管内治疗术后症状性出血转化风险预测器",
    layout="wide"
)
st.title("AIS患者血管内治疗术后症状性出血转化风险预测器")
st.markdown("### 请填写以下信息，点击预测获取风险评估结果")

# ====================== 加载模型和数据（只加载一次）======================
model = joblib.load('XGBoost.pkl')
test_dataset = pd.read_excel('data.xlsx')

# 定义特征列表（根据实际列名修改，注意 "agitation " 末尾有空格）
feature_names = [
    "age", "nihss_admit", "adl_total", "pre_apt", "post_gastric_tube",
    "sbp_baseline", "sbp_admit", "agitation ",   # 注意末尾空格
    "anc_total", "bnp_total"
]

# 检查所有特征列是否存在
missing_features = [f for f in feature_names if f not in test_dataset.columns]
if missing_features:
    st.error(f"数据文件中缺少以下特征列：{missing_features}。请检查 data.xlsx 的列名是否正确。")
    st.stop()

# 提取用于LIME的训练数据（只取特征部分，且可考虑采样以加速）
X_train_lime = test_dataset[feature_names].values
# 如果数据量很大，可以采样一部分用于LIME，例如：
# if X_train_lime.shape[0] > 1000:
#     idx = np.random.choice(X_train_lime.shape[0], 1000, replace=False)
#     X_train_lime = X_train_lime[idx]

# 初始化LIME解释器（只需一次）
lime_explainer = LimeTabularExplainer(
    training_data=X_train_lime,
    feature_names=feature_names,
    class_names=['低风险', '高风险'],
    mode='classification'
)

# ====================== 输入组件（每行两个） ======================
# 第1行
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("年龄", min_value=0.0, value=0.0, step=1.0, format="%.2f")
with col2:
    nihss_admit = st.number_input("入院NIHSS评分", min_value=0.0, value=0.0, step=0.5, format="%.2f")

# 第2行
col1, col2 = st.columns(2)
with col1:
    adl_total = st.number_input("基线自理能力评分", min_value=0.0, value=0.0, step=1.0, format="%.2f")
with col2:
    pre_apt = st.selectbox("术前是否使用抗凝抗板药物", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

# 第3行
col1, col2 = st.columns(2)
with col1:
    sbp_baseline = st.number_input("基线收缩压", min_value=0.0, value=0.0, step=1.0, format="%.2f")
with col2:
    sbp_admit = st.number_input("入院收缩压", min_value=0.0, value=0.0, step=1.0, format="%.2f")

# 第4行
col1, col2 = st.columns(2)
with col1:
    agitation = st.selectbox(
        "术后躁动情况？",
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: "无", 1: "轻度躁动", 2: "中度躁动", 3: "重度躁动"}[x]
    )
with col2:
    anc_total = st.number_input("基线中性粒细胞计数", min_value=0.0, value=0.0, step=0.1, format="%.2f")

# 第5行
col1, col2 = st.columns(2)
with col1:
    bnp_total = st.number_input("基线BNP", min_value=0.0, value=0.0, step=1.0, format="%.2f")
with col2:
    post_gastric_tube = st.selectbox("术后是否留置胃管", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

# ====================== 预测 ======================
if st.button("预测"):
    # 构建特征数组（顺序必须与 feature_names 一致）
    feature_values = [
        age, nihss_admit, adl_total, pre_apt, post_gastric_tube,
        sbp_baseline, sbp_admit, agitation, anc_total, bnp_total
    ]
    input_df = pd.DataFrame([feature_values], columns=feature_names)

    proba = model.predict_proba(input_df)[0]
    risk_prob = proba[1]  # 高风险概率

    # 根据阈值划分风险等级
    if risk_prob < 0.20:
        pred_class = "低风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于低风险。建议继续保持当前治疗方案，定期随访。"
    elif risk_prob < 0.80:
        pred_class = "中风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于中风险。建议密切观察，遵医嘱进行相关检查。"
    else:
        pred_class = "高风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于高风险。建议立即就医，加强监测和预防措施。"

    st.subheader("📊 预测结果")
    st.write(f"**预测分类**：{pred_class}")
    st.write(f"**预测概率**：{risk_prob:.2%}")

    st.subheader("💡 健康建议")
    st.write(advice)

    # ====================== LIME 解释（使用已初始化的解释器）======================
    st.subheader("🔍 LIME特征贡献解释")
    lime_exp = lime_explainer.explain_instance(
        data_row=input_df.values.flatten(),
        predict_fn=model.predict_proba,
        num_features=10
    )
    lime_html = lime_exp.as_html(show_table=True)
    components.html(lime_html, height=600, scrolling=True)