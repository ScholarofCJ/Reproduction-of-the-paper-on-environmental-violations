import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# 1. 读取数据
df = pd.read_excel('Data.xlsx')

# 2. 定义变量组
x_greed = [
    "Chairman_sex", "Chairman_age", "Chairman_oversea", "Chairman_finance",
    "Chairman_accounting", "Chairman_academic", "Chairman_environment",
    "CEO_sex", "CEO_age", "CEO_oversea", "CEO_finance", "CEO_accounting",
    "CEO_academic", "CEO_environment", "Dual", "Incentive"
]
x_opportunity = [
    "Institution_ratio", "Top1_ratio", "Foreign_ratio", "Management",
    "Disclosure", "Big4", "Analyst", "Media", "Politics", "Controls"
]
x_need = [
    "Lev", "FC", "Liq", "ROA", "ROE", "Growth", "Profit",
    "Competition", "Dividend", "Coverage"
]
x_exposure = [
    "Regulation", "Environmental_target", "Growth_target", "ESG",
    "Credit", "Inspection", "Interview", "Court", "Concern", "Investor"
]

# 3. 筛选计算 VIF 的特征 (排除控制变量)
# 根据你的需求，剔除 x_basic 以及 x_opportunity 中的 "Controls"
features_for_vif = x_greed + \
                   x_opportunity + \
                   x_need + \
                   x_exposure

# 4. 计算 VIF 的数据准备
# 必须先删除含有缺失值的行，否则 VIF 无法计算
df_vif = df[features_for_vif].dropna().astype(float)

# 为计算 VIF 添加常数项
df_vif_const = add_constant(df_vif)

# 5. 计算 VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df_vif_const.columns
vif_data["VIF"] = [variance_inflation_factor(df_vif_const.values, i) 
                   for i in range(df_vif_const.shape[1])]

# 打印结果
print("--- VIF 计算结果 (已排除控制变量) ---")
print(vif_data.sort_values(by="VIF", ascending=False))
# 3. 删除 'const' 项
vif_data = vif_data[vif_data["feature"] != "const"]

# 4. 格式化并排序
# 保留两位小数，方便阅读
vif_data["VIF"] = vif_data["VIF"].round(2)
vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)


# 5. 导出为 excel 文件
#先把文件转置一下
vif_data = vif_data.T
vif_data.to_excel(r"Tables/Appendix-C_VIF_Results.xlsx", index=False)

print("--- VIF 计算结果 (已删除常数项，已导出为 Excel) ---")
target_vars = ["ROA", "ROE","Profit","Lev"]
other_dimensions = x_greed+x_exposure+x_opportunity
for var in target_vars:
    features_for_vif = other_dimensions + [var]
    df_vif = df[features_for_vif].dropna().astype 