import pandas as pd
import matplotlib.pyplot as plt


def plot_b_spline():
    loss_data = pd.read_csv('plotting/cnp_b_loss.csv')

    # 创建子图
    plt.figure(figsize=(18, 6))

    plt.plot(loss_data['Step'][:2000], loss_data['0-1100 - train/model_loss'][:2000], label='1', color='orange')
    plt.plot(loss_data['Step'][:2000], loss_data['traj_range0.8-1.31 - train/model_loss'][:2000], label='2', color='gray')
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.title('Total Loss', fontsize=25)
    plt.legend(fontsize=18)
    # 显示图表
    plt.show()


if __name__ == '__main__':
    front_size = 20
    y_front_size = 25
    label_front_size = 15

    plot_b_spline()