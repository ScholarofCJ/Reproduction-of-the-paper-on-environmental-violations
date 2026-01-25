import pandas as pd
import json
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, classification_report
import pickle
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler


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

# 开始对模型进行优化，调整参数
param_distributions = {
    'n_estimators': range(100, 500, 50),
    'max_depth': range(2, 30),
    'num_leaves': range(20, 150, 10),
    'min_child_samples': range(5, 30, 5),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}
kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=666)
Model = RandomizedSearchCV(estimator=LGBMClassifier(random_state=123),
                           param_distributions=param_distributions, scoring='roc_auc',
                           n_jobs=-1, cv=kfold, n_iter=100, random_state=123)

# 训练模型
Model.fit(x_train, y_train)
print(Model.best_params_)
#把最优参数保存导出#
with open(r'Model\Best_param_lgbm.json', 'w') as f:
    json.dump(Model.best_params_, f)
model = Model.best_estimator_

# 保存模型
with open(r'Model\lightgbm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 预测
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

# 输出分类指标
print('模型的测试集AUC值为：', roc_auc_score(y_test, y_pred_proba))
print('模型的测试集F1分数为：', f1_score(y_test, y_pred))
print('模型的测试集精准度为：', accuracy_score(y_test, y_pred))
print('模型的测试集精确率为：', precision_score(y_test, y_pred))
print('模型的测试集召回率为：', recall_score(y_test, y_pred))
print('分类报告：\n', classification_report(y_test, y_pred))