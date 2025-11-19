# -*- coding: utf-8 -*-
"""
双精度21×21规模运行时间柱状图
"""

import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = [
    'MATLAB\n单线程', 'MATLAB\n多线程', 
    'OpenMP\n主动集单线程', 'OpenMP\n主动集双线程', 
    'OpenMP\n主动集4线程', 'OpenMP\n主动集8线程', 
    'OpenMP\n主动集16线程', 'OpenMP\n主动集32线程', 
    'OpenMP\n梯度投影单线程', 'OpenMP\n梯度投影双线程', 
    'OpenMP\n梯度投影4线程', 'OpenMP\n梯度投影8线程', 
    'OpenMP\n梯度投影16线程', 'OpenMP\n梯度投影32线程', 
    'CUDA\n主动集', 'CUDA\n梯度投影'
]
times = [
    0.32, 0.0366, 0.0174, 0.0096, 0.0078, 0.0072, 0.0080, 0.0085, 
    0.165, 0.152, 0.142, 0.137, 0.144, 0.148, 0.0686, 0.183
]

# 设置柱状图
x = np.arange(len(methods))  # 方法标签位置
width = 0.35  # 柱子宽度

fig, ax = plt.subplots(figsize=(15, 8))  # 设置图像大小
rects = ax.bar(x, times, width, label='双精度21×21')

# 添加文本标签
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# 设置坐标轴标签和标题
ax.set_xlabel('方法')
ax.set_ylabel('运行时间 (秒)')
ax.set_title('双精度21×21规模运行时间柱状图')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')  # 旋转x轴标签以便显示
ax.legend()

# 显示网格
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()