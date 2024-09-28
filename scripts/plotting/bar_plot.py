import numpy as np
import matplotlib.pyplot as plt

# 代理名称
agents = ['origin', 'sl_origin', 'cl_r', 'sl_cl_r', 'cl_line', 'sl_cl_line']

# score 和 concede 的数据
scores =   [8.5, 9.6, 10.3, 10.25, 12.3, 10.0]
concedes = [9.5, 3.7, 5.7, 4.5 ,  0.7,  5.3]

# 生成 x 轴的位置
x = np.arange(len(agents))

# 定义柱状图的宽度
width = 0.35

# 创建图形和轴
fig, ax = plt.subplots(figsize=(16, 8))

# 绘制 score 和 concede 的柱状图
bars1 = ax.bar(x - width/2, scores, width, label='Goal')
bars2 = ax.bar(x + width/2, concedes, width, label='Lose')

# 添加标签、标题和自定义 x 轴刻度
ax.set_title('Goals and loses against baseline', fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(agents, fontsize=22)

# 设置y轴刻度字体大小
ax.tick_params(axis='y', labelsize=20)

# 添加图例
ax.legend(fontsize=20)

# 显示图形
plt.show()
