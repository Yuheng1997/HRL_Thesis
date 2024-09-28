import pandas as pd
import matplotlib.pyplot as plt


def plot_cl():
    w_data = pd.read_csv('plotting/self_learn/sl_w.csv')
    l_data = pd.read_csv('plotting/self_learn/sl_l.csv')
    r_data = pd.read_csv('plotting/self_learn/sl_r.csv')

    # 创建子图
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

    # axes[0].plot(r_data['Step'][:100],
    #              r_data['Group: 16_09_cl_reward/less_line_reward/continue0909/self_learning4pop - Reward/J'][:100],
    #              label='sl_cl_reward', color='forestgreen')
    # axes[0].plot(r_data['Step'][:100],
    #              r_data['Group: 16_09_cl_reward/continue0909/against0829 - Reward/J'][:100],
    #              linestyle='--', label='cl_reward', color='forestgreen')
    #
    # axes[0].plot(r_data['Step'][:100],
    #              r_data['Group: 15_09_curriculum_line5/less_line_reward/continue0909/self_learning4pop - Reward/J'][:100] - r_data['Group: 10_09_curriculum_line5/continue0909/two_opponent/600/against_0829_baseline/0.995 - Reward/J'][:100],
    #              label='sl_cl_line', color='orange')
    # axes[0].plot(r_data['Step'][:100],r_data['Group: 10_09_curriculum_line5/continue0909/two_opponent/600/against_0829_baseline/0.995 - Reward/J'][:100],
    #              linestyle='--', label='cl_line', color='orange')
    #
    #
    # axes[0].plot(r_data['Step'][:100],
    #              r_data['Group: 12_09_continue0909/self_learning - Reward/J'][:100] - r_data['Group: 10_09_continue0909 - Reward/J'][101:].reset_index(drop=True),
    #              label='sl_origin', color='royalblue')
    # axes[0].plot(r_data['Step'][:100],r_data['Group: 10_09_continue0909 - Reward/J'][101:].reset_index(drop=True),
    #              linestyle='--', label='origin', color='royalblue')
    # axes[0].set_title('Reward', fontsize=front_size, family='Times New Roman')
    # axes[0].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    # axes[0].tick_params(axis='x', labelsize=20)
    # axes[0].tick_params(axis='y', labelsize=20)
    # axes[0].legend()

    axes[0].plot(w_data['Step'][:100],
               w_data['Group: 16_09_cl_reward/less_line_reward/continue0909/self_learning4pop - win'][:100],
               label='sl_cl_reward', color='forestgreen')
    axes[0].plot(w_data['Step'][:100],
               w_data['Group: 16_09_cl_reward/continue0909/against0829 - win'][:100],
               linestyle='--', label='cl_reward', color='forestgreen')
    axes[0].plot(w_data['Step'][:100],
               w_data['Group: 15_09_curriculum_line5/less_line_reward/continue0909/self_learning4pop - win'][:100],
               label='sl_cl_line', color='orange')
    axes[0].plot(w_data['Step'][:100],
               w_data['Group: 10_09_curriculum_line5/continue0909/two_opponent/600/against_0829_baseline/0.995 - win'][:100],
               linestyle='--', label='cl_line', color='orange')
    axes[0].plot(w_data['Step'][:100],
               w_data['Group: 12_09_continue0909/self_learning - win'][:100],
               label='sl_origin', color='royalblue')
    axes[0].plot(w_data['Step'][:100],
               w_data['Group: 10_09_continue0909 - win'][101:].reset_index(drop=True),
               linestyle='--', label='origin', color='royalblue')
    axes[0].set_title('Goals', fontsize=front_size, family='Times New Roman')
    axes[0].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[0].tick_params(axis='x', labelsize=20)
    axes[0].tick_params(axis='y', labelsize=20)


    axes[1].plot(l_data['Step'][:100],
               l_data['Group: 16_09_cl_reward/less_line_reward/continue0909/self_learning4pop - lose'][:100],
               label='sl_cl_reward', color='forestgreen')
    axes[1].plot(l_data['Step'][:100],
               l_data['Group: 16_09_cl_reward/continue0909/against0829 - lose'][:100],
               linestyle='--', label='cl_reward', color='forestgreen')
    axes[1].plot(l_data['Step'][:100],
               l_data['Group: 15_09_curriculum_line5/less_line_reward/continue0909/self_learning4pop - lose'][:100],
               label='sl_cl_line', color='orange')
    axes[1].plot(l_data['Step'][:100],
               l_data['Group: 10_09_curriculum_line5/continue0909/two_opponent/600/against_0829_baseline/0.995 - lose'][:100],
               linestyle='--', label='cl_line', color='orange')
    axes[1].plot(l_data['Step'][:100],
               l_data['Group: 12_09_continue0909/self_learning - lose'][:100],
               label='sl_origin', color='royalblue')
    axes[1].plot(l_data['Step'][:100],
              l_data['Group: 10_09_continue0909 - lose'][101:].reset_index(drop=True),
               linestyle='--', label='origin', color='royalblue')
    axes[1].set_title('Loses', fontsize=front_size, family='Times New Roman')
    axes[1].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[1].tick_params(axis='x', labelsize=20)
    axes[1].tick_params(axis='y', labelsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=18)

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