from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import shap




df = pd.read_excel('Data.xlsx')
df_property = pd.read_stata('异质性1.dta')

# 合并
df = pd.merge(df, df_property, on=['证券代码', '年份'], how='left')

# 根据是否为重污染行业分组
df1 = df[df['是否为重污染行业'] == 1]
df2 = df[df['是否为重污染行业'] == 0]

# 开始对模型进行优化，调整参数
param_distributions = {
    'n_estimators': range(100, 1000, 50),
    'max_depth': range(2, 40),
    'min_samples_leaf': range(1, 20),
    'min_samples_split': range(2, 20),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

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
# 基本面包含Size、Age和从Ind1到Ind47的变量
x_basic = ["Size", "Age"] + [f"Ind{i}" for i in range(1, 48)]
def train_and_save(df_group, save_path):
    y = df_group['EV'].astype(int)  # 确保y是整数类型
    # 合并所有特征
    x = df_group[x_greed + x_opportunity + x_need + x_exposure + x_basic].astype(float)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=666)
    # 用RandomUnderSampler方法平衡数据
    rus = RandomUnderSampler(random_state=666)
    x_train, y_train = rus.fit_resample(x_train, y_train)

    kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=666)
    model = RandomizedSearchCV(estimator=RandomForestClassifier(criterion='gini', random_state=123),
                               param_distributions=param_distributions, scoring='roc_auc',
                               n_jobs=-1, cv=kfold, n_iter=500, random_state=123)
    model.fit(x_train, y_train)
    best_model = model.best_estimator_

    with open(save_path, 'wb') as f:
        pickle.dump(best_model, f)

    return best_model, x

def calc_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return np.abs(shap_values[1]).mean(axis=0), X.columns.tolist()

def compute_importance(feat_names, shap_vals):
    df_imp = pd.DataFrame({"Feature": feat_names, "Importance": shap_vals})
    total = df_imp["Importance"].sum()
    df_imp["Importance"] = df_imp["Importance"] / total * 100
    greed_df = df_imp[df_imp["Feature"].isin(x_greed )]
    Opportunity_df = df_imp[df_imp["Feature"].isin(x_opportunity)]
    need_df = df_imp[df_imp["Feature"].isin(x_need)]
    exposure_df = df_imp[df_imp["Feature"].isin(x_exposure)]


    return df_imp, greed_df, Opportunity_df, need_df, exposure_df


# -------------------------
# 训练模型并计算SHAP
# -------------------------
model_gov, X_gov = train_and_save(df1, 'Model\ind_heavy.pkl')
model_non, X_non = train_and_save(df2, 'Model\ind_nonheavy.pkl')
