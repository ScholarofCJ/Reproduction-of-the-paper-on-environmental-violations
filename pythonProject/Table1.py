import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)
import pickle
from imblearn.under_sampling import RandomUnderSampler
import os


path = 'Model'
#判断有没有Tables文件夹，没有就创建
if not os.path.exists('Tables'):
    os.makedirs('Tables')



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


# -------------------------------
# 4. 批量读取模型，计算各种指标
# -------------------------------
results = []

for file in os.listdir(path):
    if file.endswith('.pkl'):
        model_path = os.path.join(path, file)
        print(model_path)

        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # 预测
        y_pred = model.predict(x_test)

        # 二分类一般有 predict_proba
        try:
            y_prob = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        results.append({
            'Model': file,
            'AUC': auc,
            'Accuracy': acc,
            'F1-score': f1,
            'Precision': precision,
            'Recall': recall
        })

# -------------------------------
# 5. 结果输出到 Excel
# -------------------------------
df_metrics = pd.DataFrame(results)
output_path = os.path.join('Tables', 'Table1.xlsx')
df_metrics.to_excel(output_path, index=False)
print("所有模型指标已导出到：", output_path)