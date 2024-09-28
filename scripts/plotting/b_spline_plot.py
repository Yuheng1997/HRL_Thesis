import pandas as pd
import matplotlib.pyplot as plt


def plot_b_spline():
    r_data = pd.read_csv('plotting/r_b_spline.csv')
    t_data = pd.read_csv('plotting/t_num_b_spline.csv')
    e_data = pd.read_csv('plotting/mean_beta_b_spline.csv')

    # 创建子图
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    axes[0].plot(r_data['Step'][:85], r_data['Group: 19_08_clip_bonus*5/adv_bonus0.1/three_times_terminate - Reward/J'][:85],
                 label='limit num of terminate', color='forestgreen')
    axes[0].plot(r_data['Step'][:85], r_data['Group: 18_08_clip_bonus*5/adv_bonus0.1 - Reward/J'][:85],
                 label='normal', color='orange')
    axes[0].set_title('Reward', fontsize=front_size, family='Times New Roman')
    axes[0].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[0].tick_params(axis='x', labelsize=20)
    axes[0].tick_params(axis='y', labelsize=20)
    # axes[0].legend()

    axes[1].plot(t_data['Step'][:85], t_data['Group: 19_08_clip_bonus*5/adv_bonus0.1/three_times_terminate - termination_num_by_beta'][:85],
                 label='limit num of terminate', color='forestgreen')
    axes[1].plot(t_data['Step'][:85], t_data['Group: 18_08_clip_bonus*5/adv_bonus0.1 - termination_num_by_beta'][:85],
                 label='normal', color='orange')
    axes[1].set_title('Termination num', fontsize=front_size, family='Times New Roman')
    axes[1].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[1].tick_params(axis='x', labelsize=20)
    axes[1].tick_params(axis='y', labelsize=20)

    # 绘制data_beta_mean数据
    axes[2].plot(e_data['Step'][:85], e_data['Group: 19_08_clip_bonus*5/adv_bonus0.1/three_times_terminate - Termination/mean_beta'][:85],
                 label='limit num of terminate', color='forestgreen')
    axes[2].plot(e_data['Step'][:85], e_data['Group: 18_08_clip_bonus*5/adv_bonus0.1 - Termination/mean_beta'][:85],
                 label='normal', color='orange')
    axes[2].set_title('Mean beta', fontsize=front_size, family='Times New Roman')
    axes[2].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[2].tick_params(axis='x', labelsize=20)
    axes[2].tick_params(axis='y', labelsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=label_front_size)

    # 调整子图之间的间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # 显示图表
    plt.show()


if __name__ == '__main__':
    front_size = 20
    y_front_size = 25
    label_front_size = 15

    plot_b_spline()