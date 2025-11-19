# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# 数据，使用 'x' 替换 '×'，避免编码问题
scales = ["21x21", "81x81", "111x111", "121x121", "131x131"]
percentages = [72.08, 64.04, 89.13, 89.07, 89.31]

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(scales, percentages, color='lightblue', width=0.7, edgecolor='black', linewidth=0.5)
plt.xlabel("Problem Size", fontsize=14, labelpad=10)
plt.ylabel("lsqnonneg Time Proportion (%)", fontsize=14, labelpad=10)
plt.title("Proportion of lsqnonneg Time in Total Time (Excluding csvread)", fontsize=16, pad=15)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)

# 添加数据标签
for i, v in enumerate(percentages):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)

# 保存图像
plt.savefig("lsqnonneg_proportion.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()
