from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
import pickle
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import shap
from pyecharts.charts import Pie
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid




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


def train_and_save(df_group):
    y = df_group['EV'].astype(int)  # 确保y是整数类型
    # 合并所有特征
    x = df_group[x_greed + x_opportunity + x_need + x_exposure + x_basic].astype(float)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=666)
    # 用RandomUnderSampler方法平衡数据
    rus = RandomUnderSampler(random_state=666)
    x_train, y_train = rus.fit_resample(x_train, y_train)

    return x_train


# 读取数据
df = pd.read_excel('Data.xlsx')
df_property = pd.read_stata('异质性1.dta')

# 合并
df = pd.merge(df, df_property, on=['证券代码', '年份'], how='left')


###########################按照企业产权性质#########################################
# 根据是否为国企分组
df1 = df[df['是否为国企'] == 1]
df2 = df[df['是否为国企'] == 0]

model_gov = pickle.load(open('Model\gov_state.pkl', 'rb'))
model_non = pickle.load(open('Model\gov_nonstate.pkl', 'rb'))

X_non = train_and_save(df2)
X_gov = train_and_save(df1)


shap_vals_non, feat_names = calc_shap(model_non, X_non)
shap_vals_gov, _ = calc_shap(model_gov, X_gov)

df_imp_gov, Greed_gov, Opportunity_gov, Need_gov, Exposure_gov = compute_importance(feat_names, shap_vals_gov)
df_imp_non, Greed_non, Opportunity_non, Need_non, Exposure_non = compute_importance(feat_names, shap_vals_non)


# -------------------------
# 构造两个饼图的数据
# -------------------------
name = ['Greed','Opportunity','Need','Exposure']

value_gov = [
    Greed_gov['Importance'].sum(),
    Opportunity_gov['Importance'].sum(),
    Need_gov ['Importance'].sum(),
    Exposure_gov ['Importance'].sum()
]

value_non = [
    Greed_non['Importance'].sum(),
    Opportunity_non['Importance'].sum(),
    Need_non ['Importance'].sum(),
    Exposure_non ['Importance'].sum()
]

value_gov = [round(v, 2) for v in value_gov]
value_non = [round(v, 2) for v in value_non]

# -------------------------
# 画多饼图
# -------------------------
def pie_multiple():
    pie = Pie(init_opts=opts.InitOpts(width='1200px', height='1000px'))
    my_colors = ['#B2DFDB', '#FFEB3B', '#FF7043', '#64B5F6']

    pie.add("",
            [list(z) for z in zip(name, value_gov)],
            radius=["15%", "35%"],
            center=["30%", "50%"],
            label_opts=opts.LabelOpts(formatter="{b}\n({d}%)", font_size=20))

    pie.add("",
            [list(z) for z in zip(name, value_non)],
            radius=["15%", "35%"],
            center=["70%", "50%"],
            label_opts=opts.LabelOpts(formatter="{b}\n({d}%)", font_size=20))
    # 设置全局配置和颜色
    pie.set_colors(my_colors)  # <--- 在这里指定颜色顺序

    pie.set_global_opts(
        legend_opts=opts.LegendOpts(pos_top="15%", pos_left="35%", textstyle_opts=opts.TextStyleOpts(font_size=20)),
        title_opts=opts.TitleOpts(title="", title_textstyle_opts=opts.TextStyleOpts(font_size=24))
    )
    return pie

chart = pie_multiple()
chart.render('Figures\Figure10.html')
# -------------------------
diff_df = df_imp_gov.set_index('Feature') - df_imp_non.set_index('Feature')
diff_df = diff_df.reset_index()
# 找出属于各个维度的变量，并且按照从大到小的顺序排序
diff_greed = diff_df[diff_df['Feature'].isin(x_greed)].sort_values(by='Importance', ascending=False)
diff_opportunity = diff_df[diff_df['Feature'].isin(x_opportunity)].sort_values(by='Importance', ascending=False)
diff_need = diff_df[diff_df['Feature'].isin(x_need)].sort_values(by='Importance', ascending=False)
diff_exposure = diff_df[diff_df['Feature'].isin(x_exposure)].sort_values(by='Importance', ascending=False)

# -------------------------
# 绘制三个并列柱状图
# -------------------------
bar_greed = (
    Bar()
    .add_xaxis(diff_greed['Feature'].tolist())
    .add_yaxis("Greed", diff_greed['Importance'].tolist(), itemstyle_opts=opts.ItemStyleOpts(color='#B2DFDB'))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45, font_size=8.5)),
        title_opts=opts.TitleOpts(title=""),
        legend_opts=opts.LegendOpts(pos_top="8%", pos_left="20%")
    )
)
bar_opportunity = (
    Bar()
    .add_xaxis(diff_opportunity['Feature'].tolist())
    .add_yaxis("Opportunity", diff_opportunity['Importance'].tolist(), itemstyle_opts=opts.ItemStyleOpts(color='#FFEB3B'))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45, font_size=8.5)),
        title_opts=opts.TitleOpts(title=""),
        legend_opts=opts.LegendOpts(pos_top="8%", pos_left="70%")
    )
)
bar_need = (
    Bar()
    .add_xaxis(diff_need['Feature'].tolist())
    .add_yaxis("Need", diff_need['Importance'].tolist(), itemstyle_opts=opts.ItemStyleOpts(color='#FF7043'))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-30, font_size=8.5)),
        title_opts=opts.TitleOpts(title=""),
        legend_opts=opts.LegendOpts(pos_top="58%", pos_left="20%")
    )
)
bar_exposure = (
    Bar()
    .add_xaxis(diff_exposure['Feature'].tolist())
    .add_yaxis("Exposure", diff_exposure['Importance'].tolist(), itemstyle_opts=opts.ItemStyleOpts(color='#64B5F6'))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-30, font_size=8.5)),
        title_opts=opts.TitleOpts(title=""),
        legend_opts=opts.LegendOpts(pos_top="58%", pos_left="70%")
    )
)

grid = (
    Grid()
    # 上左
    .add(
        bar_greed,
        grid_opts=opts.GridOpts(
            pos_left="5%", pos_right="55%",
            pos_top="8%", pos_bottom="62%"
        )
    )
    # 上右
    .add(
        bar_opportunity,
        grid_opts=opts.GridOpts(
            pos_left="55%", pos_right="5%",
            pos_top="8%", pos_bottom="62%"
        )
    )
    # 下左
    .add(
        bar_need,
        grid_opts=opts.GridOpts(
            pos_left="5%", pos_right="55%",
            pos_top="62%", pos_bottom="8%"
        )
    )
    # 下右
    .add(
        bar_exposure,
        grid_opts=opts.GridOpts(
            pos_left="55%", pos_right="5%",
            pos_top="62%", pos_bottom="8%"
        )
    )
    .render('Figures\Figure11.html')
)


########################################按照行业分类#######################################
df1 = df[df['是否为重污染行业'] == 1]
df2 = df[df['是否为重污染行业'] == 0]
model_pollution = pickle.load(open('Model\ind_heavy.pkl', 'rb'))
model_nonpollution = pickle.load(open('Model\ind_nonheavy.pkl', 'rb'))


X_heavy = train_and_save(df1)
X_non = train_and_save(df2)



shap_vals_non, feat_names = calc_shap(model_nonpollution, X_non)
shap_vals_heavy, _ = calc_shap(model_pollution, X_heavy)

df_imp_gov, Greed_gov, Opportunity_gov, Need_gov, Exposure_gov = compute_importance(feat_names, shap_vals_heavy)
df_imp_non, Greed_non, Opportunity_non, Need_non, Exposure_non = compute_importance(feat_names, shap_vals_non)


# -------------------------
# 构造两个饼图的数据
# -------------------------
name = ['Greed','Opportunity','Need','Exposure']

value_gov = [
    Greed_gov['Importance'].sum(),
    Opportunity_gov['Importance'].sum(),
    Need_gov ['Importance'].sum(),
    Exposure_gov ['Importance'].sum()
]

value_non = [
    Greed_non['Importance'].sum(),
    Opportunity_non['Importance'].sum(),
    Need_non ['Importance'].sum(),
    Exposure_non ['Importance'].sum()
]

value_gov = [round(v, 2) for v in value_gov]
value_non = [round(v, 2) for v in value_non]

# -------------------------
# 画多饼图
# -------------------------
def pie_multiple():
    pie = Pie(init_opts=opts.InitOpts(width='1200px', height='1000px'))
    my_colors = ['#B2DFDB', '#FFEB3B', '#FF7043', '#64B5F6']
    pie.add("",
            [list(z) for z in zip(name, value_gov)],
            radius=["15%", "35%"],
            center=["30%", "50%"],
            label_opts=opts.LabelOpts(formatter="{b}\n({d}%)", font_size=20))

    pie.add("",
            [list(z) for z in zip(name, value_non)],
            radius=["15%", "35%"],
            center=["70%", "50%"],
            label_opts=opts.LabelOpts(formatter="{b}\n({d}%)", font_size=20))
    # 设置全局配置和颜色
    pie.set_colors(my_colors)  # <--- 在这里指定颜色顺序
    pie.set_global_opts(
        legend_opts=opts.LegendOpts(pos_top="15%", pos_left="35%", textstyle_opts=opts.TextStyleOpts(font_size=20)),
        title_opts=opts.TitleOpts(title="", title_textstyle_opts=opts.TextStyleOpts(font_size=24))
    )
    return pie

chart = pie_multiple()
chart.render("Figures\Figure12.html")
# -------------------------
diff_df = df_imp_gov.set_index('Feature') - df_imp_non.set_index('Feature')
diff_df = diff_df.reset_index()
# 找出属于各个维度的变量，并且按照从大到小的顺序排序
diff_greed = diff_df[diff_df['Feature'].isin(x_greed)].sort_values(by='Importance', ascending=False)
diff_opportunity = diff_df[diff_df['Feature'].isin(x_opportunity)].sort_values(by='Importance', ascending=False)
diff_need = diff_df[diff_df['Feature'].isin(x_need)].sort_values(by='Importance', ascending=False)
diff_exposure = diff_df[diff_df['Feature'].isin(x_exposure)].sort_values(by='Importance', ascending=False)

# -------------------------
# 绘制四个并列柱状图
# -------------------------
bar_greed = (
    Bar()
    .add_xaxis(diff_greed['Feature'].tolist())
    .add_yaxis("Greed", diff_greed['Importance'].tolist(), itemstyle_opts=opts.ItemStyleOpts(color='#B2DFDB'))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45, font_size=8.5)),
        title_opts=opts.TitleOpts(title=""),
        legend_opts=opts.LegendOpts(pos_top="8%", pos_left="20%")
    )
)
bar_opportunity = (
    Bar()
    .add_xaxis(diff_opportunity['Feature'].tolist())
    .add_yaxis("Opportunity", diff_opportunity['Importance'].tolist(), itemstyle_opts=opts.ItemStyleOpts(color='#FFEB3B'))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45, font_size=8.5)),
        title_opts=opts.TitleOpts(title=""),
        legend_opts=opts.LegendOpts(pos_top="8%", pos_left="70%")
    )
)
bar_need = (
    Bar()
    .add_xaxis(diff_need['Feature'].tolist())
    .add_yaxis("Need", diff_need['Importance'].tolist(), itemstyle_opts=opts.ItemStyleOpts(color='#FF7043'))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=--30, font_size=8.5)),
        title_opts=opts.TitleOpts(title=""),
        legend_opts=opts.LegendOpts(pos_top="58%", pos_left="20%")
    )
)
bar_exposure = (
    Bar()
    .add_xaxis(diff_exposure['Feature'].tolist())
    .add_yaxis("Exposure", diff_exposure['Importance'].tolist(), itemstyle_opts=opts.ItemStyleOpts(color='#64B5F6'))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-30, font_size=8.5)),
        title_opts=opts.TitleOpts(title=""),
        legend_opts=opts.LegendOpts(pos_top="58%", pos_left="70%")
    )
)

grid = (
    Grid()
    # 上左
    .add(
        bar_greed,
        grid_opts=opts.GridOpts(
            pos_left="5%", pos_right="55%",
            pos_top="8%", pos_bottom="62%"
        )
    )
    # 上右
    .add(
        bar_opportunity,
        grid_opts=opts.GridOpts(
            pos_left="55%", pos_right="5%",
            pos_top="8%", pos_bottom="62%"
        )
    )
    # 下左
    .add(
        bar_need,
        grid_opts=opts.GridOpts(
            pos_left="5%", pos_right="55%",
            pos_top="62%", pos_bottom="8%"
        )
    )
    # 下右
    .add(
        bar_exposure,
        grid_opts=opts.GridOpts(
            pos_left="55%", pos_right="5%",
            pos_top="62%", pos_bottom="8%"
        )
    )
    .render("Figures/Figure13.html")
)





