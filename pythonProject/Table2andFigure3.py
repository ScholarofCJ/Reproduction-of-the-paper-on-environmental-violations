import pickle
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')
import matplotlib.cm as cm
import numpy as np
import plotly.express as px


model_path = r'Model\RF_model.pkl'


df = pd.read_excel('Data.xlsx')
y = df['EV'].astype(int)  # 确保y是整数类型
# Panel A: Greed (贪婪维度)
x_greed = [
    "Chairman_sex", "Chairman_age", "Chairman_oversea", "Chairman_finance",
    "Chairman_accounting", "Chairman_academic", "Chairman_environment",
    "CEO_sex", "CEO_age", "CEO_oversea", "CEO_finance", "CEO_accounting",
    "CEO_academic", "CEO_environment", "Dual", "Incentive"
]

# Panel B: Opportunity (机会维度)
x_opportunity = [
    "Institution_ratio", "Top1_ratio", "Foreign_ratio", "Management",
    "Disclosure", "Big4", "Analyst", "Media", "Politics", "Controls"
]

# Panel C: Need (需求维度)
x_need = [
    "Lev", "FC", "Liq", "ROA", "ROE", "Growth", "Profit",
    "Competition", "Dividend", "Coverage"
]

# Panel D: Exposure (风险暴露维度)
x_exposure = [
    "Regulation", "Environmental_target", "Growth_target", "ESG",
    "Credit", "Inspection", "Interview", "Court", "Concern", "Investor"
]

#基本面包含Size、Age和从Ind1到Ind47的变量
x_basic = ["Size", "Age"] + [f"Ind{i}" for i in range(1, 48)]

# 合并所有特征
x = df[x_greed + x_opportunity + x_need + x_exposure + x_basic].astype(float)
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y, random_state=666)
#用RandomUnderSampler方法平衡数据
rus = RandomUnderSampler(random_state=666)
x_train, y_train = rus.fit_resample(x_train, y_train)

# 加载模型
model = pickle.load(open(model_path, 'rb'))
# 计算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)

# 把所有变量重要性导出到一个表格中
importance_df = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': np.abs(shap_values[1]).mean(axis=0)
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


# 把四个维度的变量重要性分别导出统计到表格
greed_importance = importance_df[importance_df['Feature'].isin(x_greed)].copy()
opportunity_importance = importance_df[importance_df['Feature'].isin(x_opportunity)].copy()
need_importance = importance_df[importance_df['Feature'].isin(x_need)].copy()
exposure_importance = importance_df[importance_df['Feature'].isin(x_exposure)].copy()



# 使用 pd.concat 安全地添加总计行
greed_importance = pd.concat([greed_importance, pd.DataFrame({'Feature': ['贪婪维度总重要性'], 'Importance': [greed_importance['Importance'].sum()]})], ignore_index=True)
opportunity_importance = pd.concat([opportunity_importance, pd.DataFrame({'Feature': ['机会维度总重要性'], 'Importance': [opportunity_importance['Importance'].sum()]})], ignore_index=True)
need_importance = pd.concat([need_importance, pd.DataFrame({'Feature': ['需求维度总重要性'], 'Importance': [need_importance['Importance'].sum()]})], ignore_index=True)
exposure_importance = pd.concat([exposure_importance, pd.DataFrame({'Feature': ['风险暴露维度总重要性'], 'Importance': [exposure_importance['Importance'].sum()]})], ignore_index=True)





#################################################绘图代码#########################################
# ===================== 构造旭日图用的全量数据 =====================
sunburst_data = []

def build_sunburst_df(feature_list, dimension_name):
    df_dim = importance_df[importance_df["Feature"].isin(feature_list)].copy()
    df_dim["Dimension"] = dimension_name
    return df_dim[["Dimension", "Feature", "Importance"]]

sunburst_data.append(build_sunburst_df(x_greed, "Greed"))
sunburst_data.append(build_sunburst_df(x_opportunity, "Opportunity"))
sunburst_data.append(build_sunburst_df(x_need, "Need"))
sunburst_data.append(build_sunburst_df(x_exposure, "Exposure"))

sunburst_data = pd.concat(sunburst_data, axis=0, ignore_index=True)

print(sunburst_data.head())
# ===================== 绘制完整旭日图（维度 + 全量变量） =====================
fig = px.sunburst(
    sunburst_data,
    path=["Dimension", "Feature"],
    values="Importance",
    color="Dimension",
    color_discrete_sequence=['#B2DFDB', '#FFEB3B', '#FF7043', '#64B5F6']
)

fig.update_traces(
    textinfo="label+percent entry",
    insidetextorientation="radial",
    insidetextfont=dict(size=48)  # 调整内圈字体大小
)

fig.update_layout(
    title="SHAP Importance: Dimension → Feature",
    font=dict(family="Times New Roman", size=18),
    width=2400,
    height=2400
)

fig.show()
# ===== PDF（投稿用）=====
fig.write_image(
    r"Figures\Figure3.pdf"
)

# ===== PNG（汇报 / PPT 用）=====
fig.write_image(
    r"Figures\Figure3.png",
    scale=3
)


def get_top_n_features(feature_list, dimension_name, top_n):
    df_dim = importance_df[importance_df['Feature'].isin(feature_list)].copy()
    df_dim = df_dim.sort_values("Importance", ascending=False).head(top_n)
    df_dim["Dimension"] = dimension_name
    df_dim["TopN"] = top_n
    return df_dim
# ===================== 汇总所有 Top-N 变量重要性 =====================
topn_all_results = []
for TOP_N in [5, 7, 9]:
    greed_top = get_top_n_features(x_greed, "Greed", TOP_N)
    opportunity_top = get_top_n_features(x_opportunity, "Opportunity", TOP_N)
    need_top = get_top_n_features(x_need, "Need", TOP_N)
    exposure_top = get_top_n_features(x_exposure, "Exposure", TOP_N)

    topn_all_results.append(
        pd.concat(
            [greed_top, opportunity_top, need_top, exposure_top],
            axis=0
        )
    )
topn_all_df = pd.concat(topn_all_results, ignore_index=True)
#把topn_all_df按照Dimension	TopN分组计算Importance总和
new_topn = topn_all_df.groupby(['Dimension', 'TopN'], as_index=False)['Importance'].sum()
#再按照TopN进行分组计算占比
new_topn['Share'] = new_topn['Importance']/new_topn.groupby('TopN')['Importance'].transform('sum')
topn_all_df = new_topn[['Dimension', 'TopN', 'Share']]
#把topn_all_df透视成，以TopN和Dimension下的四个维度各自占一列
topn_all_df = topn_all_df.pivot(index='TopN', columns='Dimension', values='Share').reset_index()
#去掉Dimension列，剩下的列排序为TopN, Greed, Opportunity, Need, Exposure
topn_all_df = topn_all_df[['TopN', 'Greed', 'Opportunity', 'Need', 'Exposure']]
# ===================== 导出到 Excel（不同 Sheet） =====================
output_path = r"Tables\Table2.xlsx"

with pd.ExcelWriter(output_path) as writer:

    # Sheet 1：所有 Top-N 变量结果
    topn_all_df.to_excel(
        writer,
        sheet_name="TopN_Variable_Importance",
        index=False
    )





