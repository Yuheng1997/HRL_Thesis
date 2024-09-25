import pandas as pd
import matplotlib.pyplot as plt


def plot_success(front_size, y_front_size, label_front_size):
    data = pd.read_csv('plotting/success_rate.csv')

    plt.figure(figsize=(12, 8))

    # 绘制不同的beta值的Reward/J曲线
    plt.plot(data['Step'][:101],
             data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - success_rate'][:101],
             label='Beta 0.1', color='forestgreen')
    plt.plot(data['Step'][:101], data['Group: 14_09_fixed_beta0.5against_0829_baseline - success_rate'][:101],
             label='Beta 0.5', color='orange')
    plt.plot(data['Step'][:101],
             data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - success_rate'][:101],
             label='Beta 0.9', color='royalblue')
    plt.plot(data['Step'][:101], data['Group: 10_09_continue0909 - success_rate'][:101],
             label='Beta updating', color='indianred')
    # 添加标题和标签
    plt.title('Success rate', fontsize=front_size)
    plt.xlabel('Step', fontsize=front_size)
    plt.ylabel('success rate values', fontsize=y_front_size)

    # 显示图例
    plt.legend(fontsize=label_front_size)

    # 显示网格
    plt.grid(True)

    # 显示图表
    plt.show()

def plot_win(front_size, y_front_size, label_front_size):
    data = pd.read_csv('plotting/win.csv')

    plt.figure(figsize=(12, 8))

    # 绘制不同的beta值的Reward/J曲线
    plt.plot(data['Step'][:101],
             data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - win'][:101],
             label='Beta 0.1', color='forestgreen')
    plt.plot(data['Step'][:101], data['Group: 14_09_fixed_beta0.5against_0829_baseline - win'][:101],
             label='Beta 0.5', color='orange')
    plt.plot(data['Step'][:101],
             data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - win'][:101],
             label='Beta 0.9', color='royalblue')
    plt.plot(data['Step'][:101], data['Group: 10_09_continue0909 - win'][:101],
             label='Beta updating', color='indianred')
    # 添加标题和标签
    plt.title('Score', fontsize=front_size)
    plt.xlabel('Steps', fontsize=front_size)
    plt.ylabel('score numbers', fontsize=y_front_size)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # 显示图例
    plt.legend(fontsize=label_front_size)

    # 显示图表
    plt.show()

def plot_lose(front_size, y_front_size, label_front_size):
    data = pd.read_csv('plotting/lose.csv')

    plt.figure(figsize=(12, 8))

    # 绘制不同的beta值的Reward/J曲线
    plt.plot(data['Step'][:101],
             data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - lose'][:101],
             label='Beta 0.1', color='forestgreen')
    plt.plot(data['Step'][:101], data['Group: 14_09_fixed_beta0.5against_0829_baseline - lose'][:101],
             label='Beta 0.5', color='orange')
    plt.plot(data['Step'][:101],
             data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - lose'][:101],
             label='Beta 0.9', color='royalblue')
    plt.plot(data['Step'][:101], data['Group: 10_09_continue0909 - lose'][:101],
             label='Beta updating', color='indianred')
    # 添加标题和标签
    plt.title('Concede', fontsize=front_size)
    plt.xlabel('Step', fontsize=front_size)
    plt.ylabel('concede numbers', fontsize=y_front_size)

    # 显示图例
    plt.legend(fontsize=label_front_size)

    # 显示图表
    plt.show()

def plot_alpha(front_size, y_front_size, label_front_size):
    data = pd.read_csv('plotting/alpha.csv')

    plt.figure(figsize=(12, 8))

    # 绘制不同的beta值的Reward/J曲线
    plt.plot(data['Step'][:101],
             data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - Training/alpha'][:101],
             label='Beta 0.1', color='forestgreen')
    plt.plot(data['Step'][:101], data['Group: 14_09_fixed_beta0.5against_0829_baseline - Training/alpha'][:101],
             label='Beta 0.5', color='orange')
    plt.plot(data['Step'][:101],
             data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - Training/alpha'][:101],
             label='Beta 0.9', color='royalblue')
    plt.plot(data['Step'][:101], data['Group: 10_09_continue0909 - Training/alpha'][:101],
             label='Beta updating', color='indianred')
    # 添加标题和标签
    plt.title('Alpha', fontsize=front_size)
    plt.xlabel('Step', fontsize=front_size)
    plt.ylabel('alpha values', fontsize=y_front_size)

    # 显示图例
    plt.legend(fontsize=label_front_size)

    # 显示图表
    plt.show()

def plot_entropy(front_size, y_front_size, label_front_size):
    data = pd.read_csv('plotting/entropy.csv')

    plt.figure(figsize=(12, 8))

    # 绘制不同的beta值的Reward/J曲线
    plt.plot(data['Step'][:101],
             data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - Training/E'][:101],
             label='Beta 0.1', color='forestgreen')
    plt.plot(data['Step'][:101], data['Group: 14_09_fixed_beta0.5against_0829_baseline - Training/E'][:101],
             label='Beta 0.5', color='orange')
    plt.plot(data['Step'][:101],
             data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - Training/E'][:101],
             label='Beta 0.9', color='royalblue')
    plt.plot(data['Step'][:101], data['Group: 10_09_continue0909 - Training/E'][:101],
             label='Beta updating', color='indianred')
    # 添加标题和标签
    plt.title('Entropy', fontsize=front_size)
    plt.xlabel('Step', fontsize=front_size)
    plt.ylabel('entropy values', fontsize=y_front_size)

    # 显示图例
    plt.legend(fontsize=label_front_size)

    # 显示图表
    plt.show()

def plot_reward(front_size, y_front_size, label_front_size):
    data = pd.read_csv('plotting/discounted_r.csv')
    plt.figure(figsize=(12, 8))

    # 绘制不同的beta值的Reward/J曲线
    plt.plot(data['Step'][:101], data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - Reward/J'][:101],
             label='Beta 0.1', color='forestgreen')
    plt.plot(data['Step'][:101], data['Group: 14_09_fixed_beta0.5against_0829_baseline - Reward/J'][:101],
             label='Beta 0.5', color='orange')
    plt.plot(data['Step'][:101], data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - Reward/J'][:101],
             label='Beta 0.9', color='royalblue')
    plt.plot(data['Step'][:101], data['Group: 10_09_continue0909 - Reward/J'][:101],
                 label='Beta updating', color='indianred')
    # 添加标题和标签
    plt.title('Discounted Reward', fontsize=front_size)
    plt.xlabel('Step', fontsize=front_size)
    plt.ylabel('Reward', fontsize=y_front_size)

    # 显示图例
    plt.legend(fontsize=label_front_size)

    # 显示图表
    plt.show()

def plot_beta(front_size, y_front_size, label_front_size):
    # 加载CSV文件
    data_beta_min = pd.read_csv('plotting/wandb_export_2024-09-15T13_25_24.622+02_00.csv')
    data_beta_max = pd.read_csv('plotting/wandb_export_2024-09-15T13_59_37.494+02_00.csv')
    data_beta_mean = pd.read_csv('plotting/wandb_export_2024-09-15T13_59_29.044+02_00.csv')

    # 创建子图
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    y_min = 0
    y_max = 1

    # 绘制data_beta_min数据
    axes[0].plot(data_beta_min['Step'][:101], data_beta_min['Group: fixed beta 0.1 - Termination/min_beta'][:101], label='Beta 0.1', color='forestgreen')
    axes[0].plot(data_beta_min['Step'][:101], data_beta_min['Group: fixed beta 0.5 - Termination/min_beta'][:101], label='Beta 0.5', color='orange')
    axes[0].plot(data_beta_min['Step'][:101], data_beta_min['Group: fixed beta 0.9 - Termination/min_beta'][:101], label='Beta 0.9', color='royalblue')
    axes[0].plot(data_beta_min['Step'][:101], data_beta_min['Group: 10_09_continue0909 - Termination/min_beta'][:101], label='Updating Beta', color='indianred')
    axes[0].set_title('Minimum Beta Values', fontsize=front_size, family='Times New Roman')
    axes[0].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[0].tick_params(axis='x', labelsize=20)
    axes[0].tick_params(axis='y', labelsize=20)
    axes[0].set_ylim([y_min, y_max])

    # 绘制data_beta_max数据
    axes[1].plot(data_beta_max['Step'][:101], data_beta_max['Group: fixed beta 0.1 - Termination/max_beta'][:101], label='Beta 0.1', color='forestgreen')
    axes[1].plot(data_beta_max['Step'][:101], data_beta_max['Group: fixed beta 0.5 - Termination/max_beta'][:101], label='Beta 0.5', color='orange')
    axes[1].plot(data_beta_max['Step'][:101], data_beta_max['Group: fixed beta 0.9 - Termination/max_beta'][:101], label='Beta 0.9', color='royalblue')
    axes[1].plot(data_beta_max['Step'][:101], data_beta_max['Group: 10_09_continue0909 - Termination/max_beta'][:101], label='Updating Beta', color='indianred')
    axes[1].set_title('Maximum Beta Values', fontsize=front_size, family='Times New Roman')
    axes[1].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[1].tick_params(axis='x', labelsize=20)
    axes[1].tick_params(axis='y', labelsize=20)
    axes[1].set_ylim([y_min, y_max])

    # 绘制data_beta_mean数据
    axes[2].plot(data_beta_mean['Step'][:101], data_beta_mean['Group: fixed beta 0.1 - Termination/mean_beta'][:101], label='Beta 0.1', color='forestgreen')
    axes[2].plot(data_beta_mean['Step'][:101], data_beta_mean['Group: fixed beta 0.5 - Termination/mean_beta'][:101], label='Beta 0.5', color='orange')
    axes[2].plot(data_beta_mean['Step'][:101], data_beta_mean['Group: fixed beta 0.9 - Termination/mean_beta'][:101], label='Beta 0.9', color='royalblue')
    axes[2].plot(data_beta_mean['Step'][:101], data_beta_mean['Group: 10_09_continue0909 - Termination/mean_beta'][:101], label='Updating Beta', color='indianred')
    axes[2].set_title('Mean Beta Values', fontsize=front_size, family='Times New Roman')
    axes[2].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[2].tick_params(axis='x', labelsize=20)
    axes[2].tick_params(axis='y', labelsize=20)
    axes[2].set_ylim([y_min, y_max])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=label_front_size)

    # 调整子图之间的间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # 显示图表
    plt.show()

def plot_r_w_l_e():
    w_data = pd.read_csv('plotting/win.csv')
    l_data = pd.read_csv('plotting/lose.csv')
    r_data = pd.read_csv('plotting/discounted_r.csv')
    e_data = pd.read_csv('plotting/entropy.csv')

    # 创建子图
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 6))

    axes[0].plot(r_data['Step'][:101], r_data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - Reward/J'][:101],
                 label='Beta 0.1', color='forestgreen')
    axes[0].plot(r_data['Step'][:101], r_data['Group: 14_09_fixed_beta0.5against_0829_baseline - Reward/J'][:101],
                 label='Beta 0.5', color='orange')
    axes[0].plot(r_data['Step'][:101], r_data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - Reward/J'][:101],
                 label='Beta 0.9', color='royalblue')
    axes[0].plot(r_data['Step'][:101], r_data['Group: 10_09_continue0909 - Reward/J'][:101],
                 label='Updating Beta', color='indianred')
    axes[0].set_title('Reward', fontsize=front_size, family='Times New Roman')
    axes[0].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[0].tick_params(axis='x', labelsize=20)
    axes[0].tick_params(axis='y', labelsize=20)
    # axes[0].legend()

    axes[1].plot(w_data['Step'][:101], w_data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - win'][:101],
                 label='Beta 0.1', color='forestgreen')
    axes[1].plot(w_data['Step'][:101], w_data['Group: 14_09_fixed_beta0.5against_0829_baseline - win'][:101],
                 label='Beta 0.5', color='orange')
    axes[1].plot(w_data['Step'][:101], w_data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - win'][:101],
                 label='Beta 0.9', color='royalblue')
    axes[1].plot(w_data['Step'][:101], w_data['Group: 10_09_continue0909 - win'][:101],
                 label='Updating Beta', color='indianred')
    axes[1].set_title('Goals', fontsize=front_size, family='Times New Roman')
    axes[1].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[1].tick_params(axis='x', labelsize=20)
    axes[1].tick_params(axis='y', labelsize=20)

    # 绘制data_beta_mean数据
    axes[2].plot(l_data['Step'][:101], l_data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - lose'][:101],
                 label='Beta 0.1', color='forestgreen')
    axes[2].plot(l_data['Step'][:101], l_data['Group: 14_09_fixed_beta0.5against_0829_baseline - lose'][:101],
                 label='Beta 0.5', color='orange')
    axes[2].plot(l_data['Step'][:101], l_data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - lose'][:101],
                 label='Beta 0.9', color='royalblue')
    axes[2].plot(l_data['Step'][:101],
                 l_data['Group: 10_09_continue0909 - lose'][:101], label='Updating Beta',
                 color='indianred')
    axes[2].set_title('Loses', fontsize=front_size, family='Times New Roman')
    axes[2].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[2].tick_params(axis='x', labelsize=20)
    axes[2].tick_params(axis='y', labelsize=20)

    #
    axes[3].plot(e_data['Step'][:101], e_data['Group: 14_09_fixed_beta0.1/adv_0.1/600/against_0829_baseline/0.995 - Training/E'][:101],
                 label='Beta 0.1', color='forestgreen')
    axes[3].plot(e_data['Step'][:101], e_data['Group: 14_09_fixed_beta0.5against_0829_baseline - Training/E'][:101],
                 label='Beta 0.5', color='orange')
    axes[3].plot(e_data['Step'][:101], e_data['Group: 09_09_fixed_beta0.9/adv_0.1/600/against_0829_baseline/0.995 - Training/E'][:101],
                 label='Beta 0.9', color='royalblue')
    axes[3].plot(e_data['Step'][:101],
                 e_data['Group: 10_09_continue0909 - Training/E'][:101], label='Updating Beta',
                 color='indianred')
    axes[3].set_title('Entropy', fontsize=front_size, family='Times New Roman')
    axes[3].set_xlabel('Step', fontsize=front_size, family='Times New Roman')
    axes[3].tick_params(axis='x', labelsize=20)
    axes[3].tick_params(axis='y', labelsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=label_front_size)

    # 调整子图之间的间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # 显示图表
    plt.show()


if __name__ == '__main__':
    front_size = 20
    y_front_size = 25
    label_front_size = 15

    plot_beta(front_size, y_front_size, label_front_size)
    # plot_reward(front_size, y_front_size, label_front_size)
    # plot_entropy(front_size, y_front_size, label_front_size)
    # plot_alpha(front_size, y_front_size, label_front_size)
    # plot_win(front_size, y_front_size, label_front_size)
    # plot_lose(front_size, y_front_size, label_front_size)
    # plot_success(front_size, y_front_size, label_front_size)
    plot_r_w_l_e()