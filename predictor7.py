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
    page_title="急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器",
    layout="wide"
)
st.title("急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器")
st.markdown("### 请填写以下信息，点击预测获取风险评估结果")

# ====================== 加载模型和数据 ======================
# 请确保模型文件（如 XGBoost.pkl）与脚本在同一目录下
model = joblib.load('XGBoost.pkl')

# 加载数据（用于 LIME 解释）
test_dataset = pd.read_excel('data.xlsx')

# 显示数据列名（用于调试，部署后可以注释掉）
st.write("数据文件中的列名：", test_dataset.columns.tolist())

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

# ====================== 输入组件 ======================
age = st.number_input("年龄", min_value=0.0, value=0.0, step=1.0, format="%.2f")
nihss_admit = st.number_input("入院NIHSS评分", min_value=0.0, value=0.0, step=0.5, format="%.2f")
adl_total = st.number_input("基线自理能力评分", min_value=0.0, value=0.0, step=1.0, format="%.2f")
pre_apt = st.selectbox("术前是否使用抗凝抗板药物", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
sbp_baseline = st.number_input("基线收缩压", min_value=0.0, value=0.0, step=1.0, format="%.2f")
sbp_admit = st.number_input("入院收缩压", min_value=0.0, value=0.0, step=1.0, format="%.2f")
agitation = st.selectbox(
    "术后躁动情况？",
    options=[0, 1, 2, 3],
    format_func=lambda x: {0: "无", 1: "轻度躁动", 2: "中度躁动", 3: "重度躁动"}[x]
)
anc_total = st.number_input("基线中性粒细胞计数", min_value=0.0, value=0.0, step=0.1, format="%.2f")
bnp_total = st.number_input("基线BNP", min_value=0.0, value=0.0, step=1.0, format="%.2f")
post_gastric_tube = st.selectbox("术后是否留置胃管", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

# ====================== 预测 ======================
if st.button("预测"):
    # 构建特征数组（顺序必须与 feature_names 一致）
    feature_values = [
        age,
        nihss_admit,
        adl_total,
        pre_apt,
        post_gastric_tube,
        sbp_baseline,
        sbp_admit,
        agitation,          # 输入变量名不变
        anc_total,
        bnp_total
    ]
    input_df = pd.DataFrame([feature_values], columns=feature_names)

    # 获取高风险概率
    proba = model.predict_proba(input_df)[0]
    risk_prob = proba[1]  # 高风险概率

    # 根据阈值划分风险等级（可调整阈值）
    if risk_prob < 0.20:
        pred_class = "低风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于低风险。建议继续保持当前治疗方案，定期随访。"
    elif risk_prob < 0.80:
        pred_class = "中风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于中风险。建议密切观察，遵医嘱进行相关检查。"
    else:
        pred_class = "高风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于高风险。建议立即就医，加强监测和预防措施。"

    # 显示预测结果
    st.subheader("📊 预测结果")
    st.write(f"**预测分类**：{pred_class}")
    st.write(f"**预测概率**：{risk_prob:.2%}")

    # 显示健康建议
    st.subheader("💡 健康建议")
    st.write(advice)

    # ====================== LIME 解释 ======================
    st.subheader("🔍 LIME特征贡献解释")
    X_train_lime = test_dataset[feature_names].values
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_lime,
        feature_names=feature_names,
        class_names=['低风险', '高风险'],
        mode='classification'
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=input_df.values.flatten(),
        predict_fn=model.predict_proba,
        num_features=10
    )
    lime_html = lime_exp.as_html(show_table=True)
    components.html(lime_html, height=600, scrolling=True)