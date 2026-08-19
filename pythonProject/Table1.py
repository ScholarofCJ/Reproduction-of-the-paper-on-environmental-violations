import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, fbeta_score, precision_recall_curve
)
import pickle
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt


path = 'Model'

# 统一模型的显示名称和顺序（用于图例及结果表）
MODEL_ORDER = [
    'AdaBoost', 'DT', 'NB', 'LDA', 'LightGBM',
    'LR', 'MLP', 'RF', 'SVM', 'XGBoost'
]

MODEL_NAME_ALIASES = {
    'adaboost': 'AdaBoost',
    'ada': 'AdaBoost',
    'ab': 'AdaBoost',
    'decisiontree': 'DT',
    'dt': 'DT',
    'naivebayes': 'NB',
    'gaussiannb': 'NB',
    'nb': 'NB',
    'lineardiscriminantanalysis': 'LDA',
    'lda': 'LDA',
    'lightgbm': 'LightGBM',
    'lgbm': 'LightGBM',
    'logisticregression': 'LR',
    'logit': 'LR',
    'logistic': 'LR',
    'lr': 'LR',
    'multilayerperceptron': 'MLP',
    'mlp': 'MLP',
    'randomforest': 'RF',
    'rf': 'RF',
    'supportvectormachine': 'SVM',
    'svc': 'SVM',
    'svm': 'SVM',
    'xgboost': 'XGBoost',
    'xgb': 'XGBoost'
}


def get_model_display_name(filename):
    """将模型文件名转换成统一、简洁的显示名称。"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    normalized = re.sub(r'[^a-z0-9]', '', stem.lower())

    # 去掉文件名中常见但不影响模型类型的单词
    for word in ('classifier', 'model', 'best', 'tuned'):
        normalized = normalized.replace(word, '')

    if normalized in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[normalized]

    # 兼容如 random_forest_v1.pkl、xgboost2026.pkl 一类带后缀的文件名
    for alias in sorted(MODEL_NAME_ALIASES, key=len, reverse=True):
        if len(alias) > 2 and alias in normalized:
            return MODEL_NAME_ALIASES[alias]

    return stem


def model_sort_key(filename):
    display_name = get_model_display_name(filename)
    try:
        return MODEL_ORDER.index(display_name)
    except ValueError:
        return len(MODEL_ORDER)
#判断有没有Tables文件夹，没有就创建
if not os.path.exists('Tables'):
    os.makedirs('Tables')



df = pd.read_excel('Data.xlsx')
y = df['EV'].astype(int)  # 确保y是整数类型
#输出y中0和1的比例
print(y.value_counts(normalize=True))
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
pr_data=[]

for file in sorted(os.listdir(path), key=model_sort_key):
    if file.endswith('.pkl'):
        model_path = os.path.join(path, file)
        model_name = get_model_display_name(file)
        print(model_path)

        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # 预测
        y_pred = model.predict(x_test)

        # 二分类一般有 predict_proba
        try:
            y_prob = model.predict_proba(x_test)[:, 1]
            #计算pr曲线
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
            pr_data.append((model_name, precision_vals, recall_vals))
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        fbeta = fbeta_score(y_test, y_pred, beta=2)
        results.append({
            'Model': model_name,
            'AUC': auc,
            'Accuracy': acc,
            'F1-score': f1,
            'F2-score': fbeta,
            'Precision': precision,
            'Recall': recall
        })

# ============================================================
# 绘制并保存论文级 Precision–Recall Curve
# ============================================================

# 统一论文绘图字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(8.2, 6.2))

# ------------------------------------------------------------
# 1. 设置论文风格配色
#    使用区分度较高的配色，RF 单独突出
# ------------------------------------------------------------
model_colors = {
    'AdaBoost': '#0072B2',   # blue
    'DT':       '#E69F00',   # orange
    'NB':       '#009E73',   # green
    'LDA':      '#CC79A7',   # purple-pink
    'LightGBM': '#56B4E9',   # light blue
    'LR':       '#8C564B',   # brown
    'MLP':      '#F0E442',   # yellow
    'RF':       '#D62728',   # red
    'SVM':      '#7F7F7F',   # gray
    'XGBoost':  '#BCBD22'    # olive
}

# ------------------------------------------------------------
# 2. 先绘制普通模型，最后绘制 RF
#    防止 RF 遮挡其他模型
# ------------------------------------------------------------
normal_curves = [
    curve for curve in pr_data if curve[0] != 'RF'
]
rf_curves = [
    curve for curve in pr_data if curve[0] == 'RF'
]

for model_name, precision_vals, recall_vals in normal_curves:

    # PR 曲线按 Recall 从小到大排列
    order = np.argsort(recall_vals)
    recall_plot = np.asarray(recall_vals)[order]
    precision_plot = np.asarray(precision_vals)[order]

    ax.plot(
        recall_plot,
        precision_plot,
        label=model_name,
        color=model_colors.get(model_name, '#666666'),
        linewidth=1.6,
        alpha=0.75,
        solid_capstyle='round',
        zorder=2
    )

# RF 最后绘制并突出显示
for model_name, precision_vals, recall_vals in rf_curves:

    order = np.argsort(recall_vals)
    recall_plot = np.asarray(recall_vals)[order]
    precision_plot = np.asarray(precision_vals)[order]

    ax.plot(
        recall_plot,
        precision_plot,
        label=model_name,
        color=model_colors['RF'],
        linewidth=2.8,
        alpha=1.0,
        solid_capstyle='round',
        zorder=10
    )


# ------------------------------------------------------------
# 3. 坐标轴设置
# ------------------------------------------------------------
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(0.0, 1.05)

ax.set_xticks(np.arange(0.0, 1.01, 0.2))
ax.set_yticks(np.arange(0.0, 1.01, 0.2))

ax.set_xlabel(
    'Recall',
    fontsize=13,
    fontweight='normal',
    labelpad=7
)

ax.set_ylabel(
    'Precision',
    fontsize=13,
    fontweight='normal',
    labelpad=7
)

# 刻度字体
ax.tick_params(
    axis='both',
    which='major',
    labelsize=10.5,
    width=0.8,
    length=4
)


# ------------------------------------------------------------
# 4. 网格线：仅保留浅色水平网格
# ------------------------------------------------------------
ax.set_axisbelow(True)

ax.yaxis.grid(
    True,
    linestyle='--',
    linewidth=0.7,
    color='#D9D9D9',
    alpha=0.55
)

ax.xaxis.grid(False)


# ------------------------------------------------------------
# 5. 坐标轴边框
# ------------------------------------------------------------
for spine in ax.spines.values():
    spine.set_linewidth(0.9)
    spine.set_color('#333333')


# ------------------------------------------------------------
# 6. 不设置图内大标题
#    论文中建议通过 Figure caption 说明
# ------------------------------------------------------------
# ax.set_title(...)  # 不设置标题


# ------------------------------------------------------------
# 7. 图例
#    按预先设定的 MODEL_ORDER 排列
# ------------------------------------------------------------
handles, labels = ax.get_legend_handles_labels()

handle_by_label = dict(zip(labels, handles))

ordered_labels = [
    name for name in MODEL_ORDER
    if name in handle_by_label
]

ordered_handles = [
    handle_by_label[name]
    for name in ordered_labels
]

legend = ax.legend(
    ordered_handles,
    ordered_labels,
    loc='upper right',
    fontsize=9.2,
    frameon=True,
    framealpha=0.92,
    facecolor='white',
    edgecolor='#BFBFBF',
    handlelength=2.2,
    handletextpad=0.6,
    labelspacing=0.45,
    borderpad=0.6
)

# RF 图例加粗
for text in legend.get_texts():
    if text.get_text() == 'RF':
        text.set_fontweight('bold')


# ------------------------------------------------------------
# 8. 调整整体布局
# ------------------------------------------------------------
fig.tight_layout(pad=1.0)


# ------------------------------------------------------------
# 9. 保存高分辨率图片
# ------------------------------------------------------------
if not os.path.exists('Figures'):
    os.makedirs('Figures')

output_figure = os.path.join(
    'Figures',
    'Figure3PR_Curve.png'
)

fig.savefig(
    output_figure,
    dpi=800,
    bbox_inches='tight',
    facecolor='white'
)

# 同时保存 PDF，方便论文投稿
fig.savefig(
    os.path.join('Figures', 'PR_Curve.pdf'),
    bbox_inches='tight',
    facecolor='white'
)

plt.close(fig)

print("PR曲线已保存到：", output_figure)
print("同时保存了PDF版本：Figures/Figure3PR_Curve.pdf")




# -------------------------------
# 5. 结果输出到 Excel
# -------------------------------
df_metrics = pd.DataFrame(results)
output_path = os.path.join('Tables', 'Table1.xlsx')
df_metrics.to_excel(output_path, index=False)
print("所有模型指标已导出到：", output_path)