import matplotlib.ticker as mtick
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取数据
df = pd.read_excel('Data.xlsx')

# 确保类型正确
df['EV'] = df['EV'].astype(int)
df['Year'] = df['年份'].astype(int)

# 每年总样本数
total_by_year = df.groupby('Year').size()

# 每年违规企业数量
violation_by_year = df[df['EV'] == 1].groupby('Year').size()

# 合并
stat = pd.DataFrame({
    'Violation_Count': violation_by_year,
    'Total_Count': total_by_year
}).fillna(0)

# 计算占比
stat['Violation_Ratio'] = stat['Violation_Count'] / stat['Total_Count']

stat = stat.reset_index()
# 每年总样本数
total_by_year = df.groupby('Year').size()

# 每年违规企业数量
violation_by_year = df[df['EV'] == 1].groupby('Year').size()

# 合并
stat = pd.DataFrame({
    'Violation_Count': violation_by_year,
    'Total_Count': total_by_year
}).fillna(0)

# 计算占比
stat['Violation_Ratio'] = stat['Violation_Count'] / stat['Total_Count']

stat = stat.reset_index()

df = stat.copy()


# 2. 设置绘图风格
sns.set_theme(style="white")

# 3. 创建画布
fig, ax1 = plt.subplots(figsize=(12, 6))
# 使用清新绿 #C8E6C9
sns.barplot(x='Year', y='Violation_Count', data=df, ax=ax1, color='#C8E6C9', label='Violation Count')
ax1.set_xlabel('Year', fontsize=12)
# 坐标轴文字使用深绿色 #2E7D32 以保证清晰度
ax1.set_ylabel('Violation Count (Companies)', fontsize=12, color='#2E7D32', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#2E7D32')

# --- 绘制折线图 (Violation_Ratio) ---
# 使用柔和粉 #FFCDD2
ax2 = ax1.twinx()
sns.lineplot(x=range(len(df)), y='Violation_Ratio', data=df, ax=ax2,
             color='#FFCDD2', marker='o', markersize=8, linewidth=3, label='Violation Ratio')
# 坐标轴文字使用深粉色 #C62828
ax2.set_ylabel('Violation Ratio (%)', fontsize=12, color='#D32F2F', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#D32F2F')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# 4. 图表细节优化
plt.title("Annual Number and Proportion of China's Environmental Violations", fontsize=14, pad=20)
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
#判断有没有Figure文件夹，没有就创建
if not os.path.exists('Figures'):
    os.makedirs('Figures')
plt.savefig('Figures\Figure1.png', dpi=500)
plt.show()
#判断有没有Model文件夹，没有就创建
if not os.path.exists('Model'):
    os.makedirs('Model')