# 导入核心库
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器",
    layout="wide"
)

# 自定义CSS：放大输入标签和结果文字
st.markdown("""
<style>
    /* 三级标题样式（与输入标签大小匹配） */
    .stMarkdown h3 {
        font-size: 1.8rem;
    }
    /* 强制数字输入框和选择框的标签字体变大 */
    .stNumberInput > label, .stSelectbox > label {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
        margin-bottom: 0.3rem !important;
    }
    /* 调整输入框内文字大小 */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        font-size: 1.2rem;
    }
    /* 右侧结果区域的字体放大 */
    .result-area h3 {
        font-size: 2.0rem;
    }
    .result-area p, .result-area div {
        font-size: 1.6rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器")
st.markdown("### 请填写以下信息，点击预测获取风险评估结果")

# ====================== 加载模型和数据 ======================
model = joblib.load('XGBoost.pkl')
test_dataset = pd.read_excel('data.xlsx')

# 定义特征列表（根据实际列名修改，注意无空格）
feature_names = [
    "age", "nihss_admit", "adl_total", "pre_apt", "post_gastric_tube",
    "sbp_baseline", "sbp_admit", "agitation ",   # 如果数据中无空格，请删除此处空格
    "anc_total", "bnp_total"
]

# 检查所有特征列是否存在
missing_features = [f for f in feature_names if f not in test_dataset.columns]
if missing_features:
    st.error(f"数据文件中缺少以下特征列：{missing_features}。请检查 data.xlsx 的列名是否正确。")
    st.stop()

# ====================== 创建左右两列布局 ======================
left_col, right_col = st.columns([3, 2])  # 左侧宽3，右侧宽2，可根据需要调整

# ====================== 左侧：输入组件 ======================
with left_col:
    # 第1行：年龄、入院NIHSS评分
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    with col2:
        nihss_admit = st.number_input("入院NIHSS评分", min_value=0.0, value=0.0, step=0.5, format="%.2f")

    # 第2行：基线自理能力评分、术前抗凝药物
    col1, col2 = st.columns(2)
    with col1:
        adl_total = st.number_input("基线自理能力评分", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    with col2:
        pre_apt = st.selectbox("术前是否使用抗凝抗板药物", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

    # 第3行：基线收缩压、入院收缩压
    col1, col2 = st.columns(2)
    with col1:
        sbp_baseline = st.number_input("基线收缩压", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    with col2:
        sbp_admit = st.number_input("入院收缩压", min_value=0.0, value=0.0, step=1.0, format="%.2f")

    # 第4行：术后躁动情况、基线中性粒细胞计数
    col1, col2 = st.columns(2)
    with col1:
        agitation = st.selectbox(
            "术后躁动情况？",
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: "无", 1: "轻度躁动", 2: "中度躁动", 3: "重度躁动"}[x]
        )
    with col2:
        anc_total = st.number_input("基线中性粒细胞计数", min_value=0.0, value=0.0, step=0.1, format="%.2f")

    # 第5行：基线BNP、术后是否留置胃管
    col1, col2 = st.columns(2)
    with col1:
        bnp_total = st.number_input("基线BNP", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    with col2:
        post_gastric_tube = st.selectbox("术后是否留置胃管", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

    # 预测按钮（居中）
    left, center, right = st.columns([1, 1, 1])
    with center:
        predict_clicked = st.button("预测", type="primary", use_container_width=True)

# ====================== 右侧：预测结果区域 ======================
with right_col:
    st.markdown('<div class="result-area">', unsafe_allow_html=True)
    st.subheader("📊 预测结果")
    
    if predict_clicked:
        # 构建特征数组（顺序必须与 feature_names 一致）
        feature_values = [
            age, nihss_admit, adl_total, pre_apt, post_gastric_tube,
            sbp_baseline, sbp_admit, agitation, anc_total, bnp_total
        ]
        input_df = pd.DataFrame([feature_values], columns=feature_names)

        proba = model.predict_proba(input_df)[0]
        risk_prob = proba[1]

        if risk_prob < 0.20:
            pred_class = "低风险"
            advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于低风险。建议继续保持当前治疗方案，定期随访。"
        elif risk_prob < 0.80:
            pred_class = "中风险"
            advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于中风险。建议密切观察，遵医嘱进行相关检查。"
        else:
            pred_class = "高风险"
            advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于高风险。建议立即就医，加强监测和预防措施。"

        st.write(f"**预测分类**：{pred_class}")
        st.write(f"**预测概率**：{risk_prob:.2%}")

        st.subheader("💡 健康建议")
        st.write(advice)
    else:
        st.info("👈 请在左侧输入信息并点击预测")
    st.markdown('</div>', unsafe_allow_html=True)