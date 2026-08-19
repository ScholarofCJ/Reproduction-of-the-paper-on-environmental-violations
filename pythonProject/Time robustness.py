"""
固定滚动窗口：
训练 5 年，测试 3 年，每次滚动 2 年。

数据年份：2010-2022

最终完整窗口：
2010-2014 -> 2015-2017
2012-2016 -> 2017-2019
2014-2018 -> 2019-2021


每个不同训练年份区间：
1. 独立进行内部时间 CV 调参
2. 独立进行 RandomUnderSampler 欠采样
3. 独立训练 Random Forest
4. 在对应测试窗口计算 SHAP
5. 计算 Need / Opportunity / Exposure / Greed 四维度 SHAP 占比

最终只输出：
Training period | Testing period | Need | Opportunity | Exposure | Greed
以及 Mean 行。

其他滚动窗口组合全部不计算。
"""

from __future__ import annotations

import gc
import json
import os
import warnings
from pathlib import Path

# =============================================================================
# 0. 并行设置
# =============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import shap

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore")


# =============================================================================
# 1. 参数设置
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Data.xlsx"
TABLE_DIR = BASE_DIR / "Tables"

TARGET_COLUMN = "EV"
YEAR_COLUMN = "年份"

DATA_START_YEAR = 2010
DATA_END_YEAR = 2022

# ---------------------------------------------------------
# 固定窗口：5年训练，3年测试，每次滚动2年
# ---------------------------------------------------------
TRAIN_YEARS = 5
TEST_YEARS = 3
STEP_YEARS = 2

# ---------------------------------------------------------
# 随机种子
# ---------------------------------------------------------
RANDOM_STATE = 666
MODEL_RANDOM_STATE = 123

# ---------------------------------------------------------
# 每个训练窗口的随机森林参数搜索次数
# 与原代码保持一致
# ---------------------------------------------------------
N_ITER_SEARCH = 100

# ---------------------------------------------------------
# 并行设置
# ---------------------------------------------------------
SEARCH_N_JOBS = -1
RF_N_JOBS = -1
SEARCH_PRE_DISPATCH = "n_jobs"

# ---------------------------------------------------------
# 内部时间CV至少使用3年作为初始训练期
# 与原代码保持一致
# ---------------------------------------------------------
INNER_MIN_TRAIN_YEARS = 3

# ---------------------------------------------------------
# SHAP使用全部测试样本
# ---------------------------------------------------------
MAX_SHAP_SAMPLES = None


# =============================================================================
# 2. 四个维度的变量
# =============================================================================

X_GREED = [
    "Chairman_sex", "Chairman_age", "Chairman_oversea", "Chairman_finance",
    "Chairman_accounting", "Chairman_academic", "Chairman_environment",
    "CEO_sex", "CEO_age", "CEO_oversea", "CEO_finance", "CEO_accounting",
    "CEO_academic", "CEO_environment", "Dual", "Incentive",
]

X_OPPORTUNITY = [
    "Institution_ratio", "Top1_ratio", "Foreign_ratio", "Management",
    "Disclosure", "Big4", "Analyst", "Media", "Politics", "Controls",
]

X_NEED = [
    "Lev", "FC", "Liq", "ROA", "ROE", "Growth", "Profit",
    "Competition", "Dividend", "Coverage",
]

X_EXPOSURE = [
    "Regulation", "Environmental_target", "Growth_target", "ESG",
    "Credit", "Inspection", "Interview", "Court", "Concern", "Investor",
]

X_BASIC = [
    "Size", "Age"
] + [
    f"Ind{i}" for i in range(1, 48)
]


FEATURES = (
    X_GREED
    + X_OPPORTUNITY
    + X_NEED
    + X_EXPOSURE
    + X_BASIC
)


DIMENSIONS = {
    "Greed": X_GREED,
    "Opportunity": X_OPPORTUNITY,
    "Need": X_NEED,
    "Exposure": X_EXPOSURE,
}


# =============================================================================
# 3. 随机森林参数空间
# =============================================================================

PARAM_DISTRIBUTIONS = {
    "rf__n_estimators": range(100, 1000, 50),
    "rf__max_depth": range(2, 40),
    "rf__min_samples_leaf": range(1, 20),
    "rf__min_samples_split": range(2, 20),
    "rf__max_features": ["sqrt", "log2", None],
    "rf__bootstrap": [True, False],
}


DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "bootstrap": True,
}


# =============================================================================
# 4. 读取并检查数据
# =============================================================================

def prepare_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {DATA_PATH}。\n"
            f"请确保脚本和 Data.xlsx 位于同一目录。"
        )

    print("=" * 70)
    print("正在读取数据……")
    print("=" * 70)

    df = pd.read_excel(DATA_PATH)

    required_columns = [
        TARGET_COLUMN,
        YEAR_COLUMN,
    ] + FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Data.xlsx 缺少以下变量：\n"
            + str(missing_columns)
        )

    df = df.copy()

    # 年份
    df[YEAR_COLUMN] = pd.to_numeric(
        df[YEAR_COLUMN],
        errors="raise",
    ).astype(int)

    # 因变量
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise",
    ).astype(int)

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            f"{TARGET_COLUMN} 必须是0/1变量。"
        )

    # 特征
    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    missing_counts = df[FEATURES].isna().sum()
    missing_counts = missing_counts[
        missing_counts > 0
    ]

    if not missing_counts.empty:
        raise ValueError(
            "模型特征中存在缺失值或非数字值：\n"
            f"{missing_counts.sort_values(ascending=False).head(20)}"
        )

    # 保留2010-2022
    df = df[
        df[YEAR_COLUMN].between(
            DATA_START_YEAR,
            DATA_END_YEAR,
        )
    ].copy()

    df = df.sort_values(
        YEAR_COLUMN,
        kind="mergesort",
    ).reset_index(drop=True)

    # 检查年份是否完整
    expected_years = set(
        range(
            DATA_START_YEAR,
            DATA_END_YEAR + 1,
        )
    )

    observed_years = set(
        df[YEAR_COLUMN].unique()
    )

    missing_years = sorted(
        expected_years - observed_years
    )

    if missing_years:
        raise ValueError(
            f"数据中缺少年份：{missing_years}"
        )

    print(
        f"数据年份：{DATA_START_YEAR}-{DATA_END_YEAR}"
    )

    print(
        f"样本量：{len(df):,}"
    )

    return df


# =============================================================================
# 5. 生成固定的三个滚动窗口
# =============================================================================

def build_windows():

    windows = []

    train_start = DATA_START_YEAR
    round_number = 1

    while True:

        train_end = (
            train_start
            + TRAIN_YEARS
            - 1
        )

        test_start = train_end + 1

        test_end = (
            test_start
            + TEST_YEARS
            - 1
        )

        # 测试期超过数据范围就停止
        if test_end > DATA_END_YEAR:
            break

        windows.append({
            "Round": round_number,
            "Train_Start": train_start,
            "Train_End": train_end,
            "Test_Start": test_start,
            "Test_End": test_end,
        })

        train_start += STEP_YEARS
        round_number += 1

    return windows


# =============================================================================
# 6. 构建内部时间CV
# =============================================================================

def build_inner_time_cv(
    years,
    targets,
):

    years = np.asarray(years)
    targets = np.asarray(targets)

    unique_years = np.sort(
        np.unique(years)
    )

    splits = []

    for validation_year in unique_years[
        INNER_MIN_TRAIN_YEARS:
    ]:

        train_index = np.flatnonzero(
            years < validation_year
        )

        validation_index = np.flatnonzero(
            years == validation_year
        )

        if (
            len(train_index) == 0
            or len(validation_index) == 0
        ):
            continue

        # 训练集必须包含两个类别
        if len(
            np.unique(
                targets[train_index]
            )
        ) < 2:
            continue

        # 验证集必须包含两个类别
        if len(
            np.unique(
                targets[validation_index]
            )
        ) < 2:
            continue

        splits.append(
            (
                train_index,
                validation_index,
            )
        )

    return splits


# =============================================================================
# 7. 对当前训练窗口进行调参
# =============================================================================

def tune_current_training_window(
    x_train,
    y_train,
    train_year_values,
    seed,
):

    cv_splits = build_inner_time_cv(
        train_year_values.to_numpy(),
        y_train.to_numpy(),
    )

    if len(cv_splits) < 2:

        print(
            "  警告：内部时间CV少于2折，"
            "使用默认参数。"
        )

        return (
            DEFAULT_PARAMS.copy(),
            None,
            len(cv_splits),
        )

    pipeline = Pipeline([
        (
            "under_sampler",
            RandomUnderSampler(
                random_state=seed
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                criterion="gini",
                random_state=seed,
                n_jobs=1,
            ),
        ),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        scoring="roc_auc",
        n_jobs=SEARCH_N_JOBS,
        pre_dispatch=SEARCH_PRE_DISPATCH,
        cv=cv_splits,
        n_iter=N_ITER_SEARCH,
        random_state=seed,
        refit=False,
        error_score=np.nan,
        verbose=0,
    )

    search.fit(
        x_train,
        y_train,
    )

    if not np.isfinite(
        search.best_score_
    ):

        print(
            "  警告：调参结果全部无效，"
            "使用默认参数。"
        )

        return (
            DEFAULT_PARAMS.copy(),
            None,
            len(cv_splits),
        )

    best_params = {
        key.replace(
            "rf__",
            ""
        ): value
        for key, value
        in search.best_params_.items()
    }

    best_score = float(
        search.best_score_
    )

    del search
    del pipeline

    gc.collect()

    return (
        best_params,
        best_score,
        len(cv_splits),
    )


# =============================================================================
# 8. 训练最终随机森林
# =============================================================================

def fit_current_training_window(
    x_train,
    y_train,
    best_params,
    seed,
):

    sampler = RandomUnderSampler(
        random_state=seed
    )

    x_balanced, y_balanced = (
        sampler.fit_resample(
            x_train,
            y_train,
        )
    )

    model = RandomForestClassifier(
        criterion="gini",
        random_state=seed,
        n_jobs=RF_N_JOBS,
        **best_params,
    )

    model.fit(
        x_balanced,
        y_balanced,
    )

    return (
        model,
        len(y_balanced),
    )


# =============================================================================
# 9. 计算正类SHAP
# =============================================================================

def positive_class_shap(
    model,
    x_values,
):

    explainer = shap.TreeExplainer(
        model
    )

    raw_values = explainer.shap_values(
        x_values,
        check_additivity=False,
    )

    if isinstance(
        raw_values,
        list,
    ):

        values = np.asarray(
            raw_values[1]
            if len(raw_values) > 1
            else raw_values[0]
        )

    else:

        values = np.asarray(
            raw_values
        )

        if values.ndim == 3:

            if (
                values.shape[:2]
                != x_values.shape
            ):
                raise ValueError(
                    f"无法识别SHAP数组形状："
                    f"{values.shape}"
                )

            values = (
                values[:, :, 1]
                if values.shape[2] > 1
                else values[:, :, 0]
            )

    if values.shape != x_values.shape:

        raise ValueError(
            f"SHAP形状 {values.shape} "
            f"与输入形状 "
            f"{x_values.shape} 不一致。"
        )

    return values


# =============================================================================
# 10. 计算四个维度的SHAP占比
# =============================================================================

def dimension_shares_from_shap(
    shap_values,
):

    # 每个变量的平均绝对SHAP
    feature_magnitude = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    magnitude_by_feature = dict(
        zip(
            FEATURES,
            feature_magnitude,
        )
    )

    # 四个维度的SHAP贡献
    dimension_magnitude = {

        dimension: sum(
            magnitude_by_feature[
                feature
            ]
            for feature in feature_list
        )

        for dimension, feature_list
        in DIMENSIONS.items()
    }

    # 注意：
    # X_BASIC仍然进入RF模型，
    # 但不进入Need/Opportunity/Exposure/Greed
    # 四维度内部占比的分母。
    total = sum(
        dimension_magnitude.values()
    )

    if (
        not np.isfinite(total)
        or total <= 0
    ):
        raise ValueError(
            "四维度SHAP贡献总量无效。"
        )

    shares = {

        dimension:
        magnitude / total

        for dimension, magnitude
        in dimension_magnitude.items()
    }

    return shares


# =============================================================================
# 11. 单个训练-测试窗口
# =============================================================================

def process_one_window(
    df,
    window,
):

    train_start = window[
        "Train_Start"
    ]

    train_end = window[
        "Train_End"
    ]

    test_start = window[
        "Test_Start"
    ]

    test_end = window[
        "Test_End"
    ]

    print()
    print("=" * 70)
    print(
        f"训练期：{train_start}-{train_end}"
    )
    print(
        f"测试期：{test_start}-{test_end}"
    )
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 训练数据
    # -------------------------------------------------------------------------

    train_mask = df[
        YEAR_COLUMN
    ].between(
        train_start,
        train_end,
    )

    x_train = df.loc[
        train_mask,
        FEATURES,
    ]

    y_train = df.loc[
        train_mask,
        TARGET_COLUMN,
    ]

    train_year_values = df.loc[
        train_mask,
        YEAR_COLUMN,
    ]

    # 检查训练年份
    expected_train_years = set(
        range(
            train_start,
            train_end + 1,
        )
    )

    observed_train_years = set(
        train_year_values.unique()
    )

    missing_train_years = sorted(
        expected_train_years
        - observed_train_years
    )

    if missing_train_years:
        raise ValueError(
            f"训练窗口 "
            f"{train_start}-{train_end} "
            f"缺少年份："
            f"{missing_train_years}"
        )

    if len(
        np.unique(y_train)
    ) < 2:

        raise ValueError(
            f"训练窗口 "
            f"{train_start}-{train_end} "
            f"不足两个类别。"
        )

    # -------------------------------------------------------------------------
    # 当前训练窗口独立调参
    # -------------------------------------------------------------------------

    seed = (
        MODEL_RANDOM_STATE
        + train_start * 100
        + train_end
    ) % (2**32 - 1)

    print(
        "正在进行当前训练窗口的时间序列调参……"
    )

    (
        best_params,
        tuning_score,
        inner_folds,
    ) = tune_current_training_window(
        x_train,
        y_train,
        train_year_values,
        seed,
    )

    print(
        "最优参数：",
        best_params,
    )

    print(
        "内部CV AUC：",
        tuning_score,
    )

    # -------------------------------------------------------------------------
    # 用当前训练窗口重新欠采样并训练RF
    # -------------------------------------------------------------------------

    print(
        "正在训练最终随机森林……"
    )

    (
        model,
        balanced_size,
    ) = fit_current_training_window(
        x_train,
        y_train,
        best_params,
        seed,
    )

    # -------------------------------------------------------------------------
    # 测试数据
    # -------------------------------------------------------------------------

    test_mask = df[
        YEAR_COLUMN
    ].between(
        test_start,
        test_end,
    )

    x_test = df.loc[
        test_mask,
        FEATURES,
    ]

    if len(x_test) == 0:
        raise ValueError(
            f"测试窗口 "
            f"{test_start}-{test_end} "
            f"为空。"
        )

    # -------------------------------------------------------------------------
    # SHAP样本
    # -------------------------------------------------------------------------

    if (
        MAX_SHAP_SAMPLES is None
        or len(x_test)
        <= MAX_SHAP_SAMPLES
    ):

        x_shap = x_test

    else:

        x_shap = x_test.sample(
            MAX_SHAP_SAMPLES,
            random_state=(
                seed + test_start
            ),
        )

    print(
        f"正在计算SHAP……"
        f"测试样本={len(x_test):,}，"
        f"SHAP样本={len(x_shap):,}"
    )

    shap_values = positive_class_shap(
        model,
        x_shap,
    )

    shares = (
        dimension_shares_from_shap(
            shap_values
        )
    )

    del model
    gc.collect()

    # -------------------------------------------------------------------------
    # 输出
    # -------------------------------------------------------------------------

    result = {

        "Training period":
            f"{train_start}–{train_end}",

        "Testing period":
            f"{test_start}–{test_end}",

        "Need":
            shares["Need"] * 100,

        "Opportunity":
            shares["Opportunity"] * 100,

        "Exposure":
            shares["Exposure"] * 100,

        "Greed":
            shares["Greed"] * 100,
    }

    return result


# =============================================================================
# 12. 主程序
# =============================================================================

def main():

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 读取数据
    # -------------------------------------------------------------------------

    df = prepare_data()

    # -------------------------------------------------------------------------
    # 生成固定窗口
    # -------------------------------------------------------------------------

    windows = build_windows()

    print()
    print("=" * 70)
    print("本次仅运行以下滚动窗口：")
    print("=" * 70)

    for window in windows:

        print(
            f"训练："
            f"{window['Train_Start']}-"
            f"{window['Train_End']}    "
            f"测试："
            f"{window['Test_Start']}-"
            f"{window['Test_End']}"
        )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # 依次计算三个窗口
    # -------------------------------------------------------------------------

    results = []

    for window in windows:

        result = process_one_window(
            df,
            window,
        )

        results.append(
            result
        )

    # -------------------------------------------------------------------------
    # 构造最终表格
    # -------------------------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    # -------------------------------------------------------------------------
    # Mean行
    # -------------------------------------------------------------------------

    mean_row = {

        "Training period":
            "Mean",

        "Testing period":
            "/",

        "Need":
            result_df["Need"].mean(),

        "Opportunity":
            result_df[
                "Opportunity"
            ].mean(),

        "Exposure":
            result_df[
                "Exposure"
            ].mean(),

        "Greed":
            result_df["Greed"].mean(),
    }

    result_df = pd.concat(
        [
            result_df,
            pd.DataFrame(
                [mean_row]
            ),
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # 保留一位小数
    # -------------------------------------------------------------------------

    for column in [
        "Need",
        "Opportunity",
        "Exposure",
        "Greed",
    ]:

        result_df[column] = (
            result_df[column]
            .round(1)
        )

    # -------------------------------------------------------------------------
    # Excel中显示为百分比
    # -------------------------------------------------------------------------

    display_df = result_df.copy()

    for column in [
        "Need",
        "Opportunity",
        "Exposure",
        "Greed",
    ]:

        display_df[column] = (
            display_df[column]
            .map(
                lambda x:
                f"{x:.1f}%"
            )
        )

    # -------------------------------------------------------------------------
    # 打印最终表格
    # -------------------------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("最终结果")
    print("=" * 70)

    print(
        display_df.to_string(
            index=False
        )
    )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # 保存Excel
    # -------------------------------------------------------------------------

    output_path = (
        TABLE_DIR
        / "Table3.xlsx"
    )

    display_df.to_excel(
        output_path,
        index=False,
    )

    print()
    print(
        "结果已保存："
        f"{output_path}"
    )


# =============================================================================
# 13. 执行
# =============================================================================

if __name__ == "__main__":
    main()