import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import pickle
import json
import numpy as np

# =====================
# 1. 读取数据
# =====================
df = pd.read_excel('Data.xlsx')

y = df['EV'].astype(int)  # 0/1 因变量

# =====================
# 2. 特征分组
# =====================
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

x_basic = ["Size", "Age"] + [f"Ind{i}" for i in range(1, 48)]

X = df[x_greed + x_opportunity + x_need + x_exposure + x_basic].astype(float)

# =====================
# 3. 划分训练 / 测试集
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=666
)

# 欠采样
rus = RandomUnderSampler(random_state=666)
X_train, y_train = rus.fit_resample(X_train, y_train)

# =====================
# 4. Naive Bayes + 参数搜索
# =====================
param_distributions = {
    "var_smoothing": np.logspace(-12, -6, 20)
}

kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=666)

Model = RandomizedSearchCV(
    estimator=GaussianNB(),
    param_distributions=param_distributions,
    scoring="roc_auc",
    n_iter=30,
    cv=kfold,
    n_jobs=-1,
    random_state=123
)

# =====================
# 5. 训练模型
# =====================
Model.fit(X_train, y_train)

print("最优参数：", Model.best_params_)

with open(r"Model\Best_param_GaussianNB.json", "w") as f:
    json.dump(Model.best_params_, f)

model = Model.best_estimator_

# 保存模型
with open(r"Model\GaussianNB_model.pkl", "wb") as f:
    pickle.dump(model, f)

# =====================
# 6. 预测与评估
# =====================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("AUC:", roc_auc_score(y_test, y_pred_proba))
print("F1:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))
