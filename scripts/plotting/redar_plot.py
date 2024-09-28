import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 定义各个维度的标签
labels = ['Serve','Attack', 'Prepare', 'Defend']
num_vars = len(labels)

# 生成每个 Agent 的数据
# agent_origin =     [1,  450/477, 1-21/30, 1 - 873/1095]
# agent_sl_origin =  [1,  449/477, 1-7/30,  1 - 753/1095]
#
# agent_cl_r =       [1,  477/477, 1-13/30, 1 - 1052/1095]
# agent_sl_cl_r =    [1,  446/477, 1-7/30,  1 - 1095/1095]
#
# agent_cl_line =    [1,  405/477, 0,       1 - 897/1095]
# agent_sl_cl_line = [1,  475/477, 1-9/30,  1 - 711/1095]

agent_origin =     [6, 6, 4, 5]
agent_sl_origin =  [6, 6, 6, 6]

agent_cl_r =       [6, 6, 5, 3]
agent_sl_cl_r =    [6, 4, 6, 3]

agent_cl_line =    [6, 5, 2, 4]
agent_sl_cl_line = [6, 6, 6, 6]

# 计算雷达图的角度
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# 使雷达图闭合
agent_origin += agent_origin[:1]
agent_sl_origin += agent_sl_origin[:1]
agent_cl_r += agent_cl_r[:1]
agent_sl_cl_r += agent_sl_cl_r[:1]
agent_cl_line += agent_cl_line[:1]
agent_sl_cl_line += agent_sl_cl_line[:1]
angles += angles[:1]

def plot_redar(agent_A, agent_B, agent_C, agent_D, agent_E, agent_F):
    # 开始绘制图形
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, agent_A,  alpha=0.25)
    ax.plot(angles, agent_A,  linewidth=2, label='origin')

    ax.fill(angles, agent_B, alpha=0.25)
    ax.plot(angles, agent_B, linewidth=2, label='sl_origin')

    # 画 Agent C 的雷达图
    ax.fill(angles, agent_C, alpha=0.25)
    ax.plot(angles, agent_C, linewidth=2, label='cl_r')

    ax.fill(angles, agent_D,  alpha=0.25)
    ax.plot(angles, agent_D,  linewidth=2, label='sl_cl_r')

    ax.fill(angles, agent_E,  alpha=0.25)
    ax.plot(angles, agent_E,  linewidth=2, label='cl_line')

    ax.fill(angles, agent_F,  alpha=0.25)
    ax.plot(angles, agent_F,  linewidth=2, label='sl_cl_line')

    # 添加各个维度的标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=20)

    ax.tick_params(axis='y', labelsize=20)

    # 添加图例
    plt.legend(loc='upper right', fontsize=15)

    # 显示图形
    plt.show()


if __name__ == '__main__':
    plot_redar(agent_origin, agent_sl_origin, agent_cl_r, agent_sl_cl_r, agent_cl_line, agent_sl_cl_line)