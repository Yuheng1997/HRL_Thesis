import pandas as pd
import matplotlib.pyplot as plt


def plot_cl():
    w_data = pd.read_csv('plotting/hrl_goal.csv')
    l_data = pd.read_csv('plotting/hrl_lose.csv')
    r_data = pd.read_csv('plotting/hrl_r.csv')

    # 创建子图
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    axes[0].plot(r_data['Step'][:100],
                 r_data['Group: 19_09_atacom_fix - Reward/J'][:100],
                 label='Non hierarchical', color='forestgreen')
    axes[0].plot(r_data['Step'][:100], r_data['Group: 19_09_atacom - Reward/J'][:100],
                 label='Hierarchical', color='orange')
    axes[0].set_title('Reward', fontsize=front_size, family='Times New Roman')
    axes[0].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[0].tick_params(axis='x', labelsize=20)
    axes[0].tick_params(axis='y', labelsize=20)
    axes[0].legend()

    axes[1].plot(w_data['Step'][:100],
                 w_data['Group: 19_09_atacom_fix - win'][:100],
                 label='Non hierarchical', color='forestgreen')
    axes[1].plot(w_data['Step'][:100], w_data['Group: 19_09_atacom - win'][:100],
                 label='Hierarchical', color='orange')
    axes[1].set_title('Scores', fontsize=front_size, family='Times New Roman')
    axes[1].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[1].tick_params(axis='x', labelsize=20)
    axes[1].tick_params(axis='y', labelsize=20)

    # 绘制data_beta_mean数据
    axes[2].plot(l_data['Step'][:100],
                 l_data['Group: 19_09_atacom_fix - lose'][:100],
                 label='Non hierarchical', color='forestgreen')
    axes[2].plot(l_data['Step'][:100], l_data['Group: 19_09_atacom - lose'][:100],
                 label='Hierarchical', color='orange')
    axes[2].set_title('loses', fontsize=front_size, family='Times New Roman')
    axes[2].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[2].tick_params(axis='x', labelsize=20)
    axes[2].tick_params(axis='y', labelsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=18)

    # 调整子图之间的间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # 显示图表
    plt.show()



if __name__ == '__main__':
    front_size = 20
    y_front_size = 25
    label_front_size = 15

    plot_cl()