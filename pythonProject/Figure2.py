import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

def draw_cv_diagram(n_splits=6):
    # ========== 全局绘图参数 ==========
    mpl.rcParams['font.family'] = 'Times New Roman'  # 论文常用字体
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(
        figsize=(10, 6),
        dpi=150,                 # 屏幕显示更清晰
        constrained_layout=True  # 自动优化布局
    )

    # 颜色（低饱和、论文友好）
    color_train = '#C8E6C9'
    color_val = '#FFCDD2'

    # ========== 绘制交叉验证矩阵 ==========
    for i in range(n_splits):
        for j in range(n_splits):
            color = color_val if i == j else color_train

            rect = patches.Rectangle(
                (j, n_splits - i - 1),  # 整数对齐，避免模糊
                1.0, 1.0,
                linewidth=0.6,          # 边框细一点
                edgecolor='0.3',        # 深灰色，比纯黑柔和
                facecolor=color
            )
            ax.add_patch(rect)

    # ========== 坐标轴设置 ==========
    ax.set_xlim(0, n_splits)
    ax.set_ylim(0, n_splits)

    ax.set_xticks([i + 0.5 for i in range(n_splits)])
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_splits)], fontsize=12)

    ax.set_yticks([n_splits - i - 0.5 for i in range(n_splits)])
    ax.set_yticklabels([f"Split {i+1}" for i in range(n_splits)], fontsize=12)

    # 移除坐标轴边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='both', length=0)

    # ========== 图例 ==========
    train_patch = patches.Patch(color=color_train, label='Training Set')
    val_patch = patches.Patch(color=color_val, label='Validation Set')

    ax.legend(
        handles=[train_patch, val_patch],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=12
    )


    # ========== 矢量格式保存 ==========
    plt.savefig("Figures\Figure2.pdf", bbox_inches='tight')

    plt.savefig("Figures\Figure2.png", dpi=500, bbox_inches='tight')

    # plt.show()

draw_cv_diagram(6)
