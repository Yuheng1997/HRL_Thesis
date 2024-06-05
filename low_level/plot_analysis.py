from differentiable_robot_model import DifferentiableRobotModel
import scipy.optimize._linprog as _linprog
from generate_hitting_data import *
from utils.constants import Limits
import matplotlib.pyplot as plt
from config import Config
import numpy as np
import torch
import os


class PlotDataLoss:
    def __init__(self):
        self.huber = torch.nn.HuberLoss(reduction='none')
        self.dt = 0.1
        self.start_point = True
        # self.path = os.path.join(os.path.abspath(os.getcwd()), "second_replanning_data_100000.tsv")
        self.path = os.path.join(os.path.abspath(os.getcwd()), "first_replanning_data_100000.tsv")
        # self.path = os.path.join(os.path.abspath(os.getcwd()), "hitting_data_40000.tsv")
        # path = os.path.join(os.path.abspath(os.getcwd()), "datasets/train/data.tsv")
        # path = os.path.join(os.path.abspath(os.getcwd()), "datasets/replan_train/data.tsv")

    def compute_loss(self, loss_values, limits):
        loss_ = torch.relu(torch.abs(loss_values) - limits)
        loss_ = self.huber(loss_, torch.zeros_like(loss_))
        return loss_ * self.dt

    def sample_configuration(self, start_point=True, device='cpu', path=None):
        if path is None:
            dataset_path = Config.data.replan_path
        else:
            dataset_path = self.path

        data = np.loadtxt(dataset_path, delimiter='\t').astype(np.float32)

        print(len(data))
        total_length = len(data)

        train_data = torch.from_numpy(data[:total_length]).to(device)

        if start_point:
            q = train_data[:, :7]
            q_dot = train_data[:, 7:14]
            q_ddot = train_data[:, 14:21]
        else:
            q = train_data[:, 21:28]
            q_dot = train_data[:, 28:35]
            q_ddot = train_data[:, 35:42]

        hit_pos = train_data[:, 42:]

        return q, q_dot, q_ddot, hit_pos, train_data

    def obstacle_violation(self, hit_pos, position):
        ee_pos = position
        dist_from_puck = np.linalg.norm(hit_pos - ee_pos[:, :2], axis=1)
        puck_loss = torch.relu(torch.tensor(0.0798) - dist_from_puck)
        return puck_loss

    def boundary_violation(self, position):
        ee_pos = position
        huber_along_path = lambda x: self.dt * self.huber(x, torch.zeros_like(x))
        relu_huber_along_path = lambda x: huber_along_path(torch.relu(x))
        x_b = torch.tensor([0.58415, 1.51])
        y_b = torch.tensor([-0.47085, 0.47085])
        z = torch.tensor(0.1645)
        x_loss = relu_huber_along_path(x_b[0] - ee_pos[:, 0]) + relu_huber_along_path(ee_pos[:, 0] - x_b[1])
        y_loss = relu_huber_along_path(y_b[0] - ee_pos[:, 1]) + relu_huber_along_path(ee_pos[:, 1] - y_b[1])
        z_loss = relu_huber_along_path(z - ee_pos[:, 2]) + relu_huber_along_path(ee_pos[:, 2] - z)
        return x_loss, y_loss, z_loss

    def plot_curv(self, data, title):
        x = torch.linspace(0, len(data), len(data))
        y = data
        x_np = x.numpy()
        y_np = y.numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(x_np, y_np)
        plt.title(title)
        plt.xlabel('point')
        plt.ylabel('loss_value')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_curve_final_point/' + title + '.PNG')
        plt.show()

    def plot_histogram(self, data, name, start_point=True):
        bins = [0, 0.00001, 0.001, 0.1, 1, np.inf]
        counts, _ = np.histogram(data, bins=bins)

        fig, ax = plt.subplots()
        bars = ax.bar(range(len(counts)), counts, tick_label=[f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)])
        ax.set_xlabel('range')
        ax.set_ylabel('number')
        ax.set_title(f'{name}')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')
        if start_point:
            plt.savefig('loss_histogram_start_point/' + name + '_in_traj_sampled' + '.PNG')
        else:
            # plt.savefig('loss_histogram_final_point/' + name + '_in_hitting_data' + '.PNG')
            plt.savefig('loss_histogram_final_point/' + name + 'data' + '.PNG')
        plt.show()

    def plot_loss(self):
        manipulator = DifferentiableRobotModel(urdf_path=Config.manipulator.urdf_file,
                                                   tensor_args={'device': 'cpu', 'dtype': torch.float32})

        print("Data path: ", self.path)
        qs, q_dots, q_ddots, hit_poses, train_data = self.sample_configuration(start_point=self.start_point, path=self.path)


        q_limits = Limits.q7
        q_dot_limits = Limits.q_dot7
        q_ddot_limits = Limits.q_ddot7
        q_dddot_limits = Limits.q_dddot7

        # for _ in range():
        q_loss = torch.sum(self.compute_loss(qs, q_limits), dim=1)
        q_dot_loss = torch.sum(self.compute_loss(q_dots, q_dot_limits), dim=1)
        q_ddot_loss = torch.sum(self.compute_loss(q_ddots, q_ddot_limits), dim=1)
        # q_dddot_loss = compute_loss(q_dddot, q_dddot_limits)


        batch_size, n_dof = qs.shape
        q_reshaped = torch.reshape(qs, (batch_size, n_dof))
        positions = manipulator.my_compute_forward_kinematics(q_reshaped, link_name='iiwa_1/striker_tip')
        # positions = torch.reshape(positions, (batch_size, 1, 3))

        obstacle_loss = self.obstacle_violation(hit_pos=hit_poses, position=positions)
        _x_loss, _y_loss, _z_loss = self.boundary_violation(position=positions)


        max_index = int(torch.argmax(q_loss))
        print('max_index', max_index)
        print('max_q', qs[max_index])
        print('loss', q_loss[max_index])
        # print('train_data', train_data[max_index])

        # plot_curv(q_loss, 'q_loss')
        # plot_curv(q_dot_loss, 'q_dot_loss')
        # plot_curv(q_ddot_loss, 'q_ddot_loss')
        # plot_curv(obstacle_loss, 'obstacle_loss')
        # plot_curv(_x_loss, '_x_loss')
        # plot_curv(_y_loss, '_y_loss')
        # plot_curv(_z_loss, '_z_loss')

        self.plot_histogram(q_loss, 'q_loss', start_point=self.start_point)
        self.plot_histogram(q_dot_loss, 'q_dot_loss', start_point=self.start_point)
        self.plot_histogram(q_ddot_loss, 'q_ddot_loss', start_point=self.start_point)
        self.plot_histogram(obstacle_loss, 'obstacle_loss', start_point=self.start_point)
        self.plot_histogram(_x_loss, '_x_loss', start_point=self.start_point)
        self.plot_histogram(_y_loss, '_y_loss', start_point=self.start_point)
        self.plot_histogram(_z_loss, '_z_loss', start_point=self.start_point)


class CompareMaxVel:
    def __init__(self):
        self.path = os.path.join(os.path.abspath(os.getcwd()), "pos_and_vel_data_1000.tsv")
        self.err = []

    def sample_pos_and_direction(self, device='cpu'):
        data = np.loadtxt(self.path, delimiter='\t').astype(np.float32)

        print(len(data))
        total_length = len(data)
        _data = torch.from_numpy(data[:total_length]).to(device)
        hit_directions = _data[:, :2]
        hit_poses = _data[:, 2:]

        return hit_directions, hit_poses

    def optimal_gain_err(self, hit_dir, hit_pos):
        v = hit_dir
        success_inv, initial_q = inverse_kinematics(robot_model, robot_data, hit_pos)
        dq_min = -0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
        dq_max = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])

        def FKp(q):
            return forward_kinematics(robot_model, robot_data, q)[0]

        def hitting_objective(q, v):
            Jp = jacobian(robot_model, robot_data, q)[:3]
            manipulability = np.linalg.norm(np.dot(v, Jp))
            return -manipulability

        def constraint_ineq(q):
            return dq_max - np.abs(q)

        constraints = [{'type': 'eq', 'fun': lambda q: FKp(q) - hit_pos},
                       {'type': 'ineq', 'fun': constraint_ineq}]
        result_q = minimize(hitting_objective, initial_q, args=(v,), constraints=constraints)
        if not result_q.success:
            optimal_q = result_q.x
        else:
            optimal_q = initial_q

        # gain v
        jacb_star = jacobian(robot_model, robot_data, optimal_q)[:3]
        qdf = np.linalg.lstsq(jacb_star, v, rcond=None)[0]
        qdf[np.abs(qdf) < 1e-8] = 1e-8
        max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
        gain_dq = max_gain * qdf
        v_gain = jacb_star @ gain_dq

        # optimal v
        vv = v.T.dot(v)
        c = [-vv, 0, 0, 0, 0, 0, 0, 0]
        J_dag = np.linalg.pinv(jacb_star)
        _JJ = J_dag.dot(jacb_star)
        N = np.eye(*_JJ.shape) - _JJ

        J_dag_v = J_dag
        J_dag_v = J_dag_v @ v

        A = np.c_[J_dag_v, N]
        A = np.r_[A, -A]
        _dq_max = 0.999 * dq_max
        _dq_min = 0.999 * dq_min
        _ub = np.c_[[*_dq_max, *(-_dq_min)]]
        result_eta = linprog(c=c, A_ub=A, b_ub=_ub, method='highs', x0=[0, 0, 0, 0, 0, 0, 0, 0], bounds=((0, None),
                             (dq_min[0], dq_max[0]), (dq_min[1], dq_max[1]), (dq_min[2], dq_max[2]), (dq_min[3], dq_max[3]),
                             (dq_min[4], dq_max[4]), (dq_min[5], dq_max[5]), (dq_min[6], dq_max[6])))

        if result_eta.success:
            optimal_eta = result_eta.x[0]
            optimal_alpha = result_eta.x[1:]
            optimal_v = optimal_eta * v
            _dq = J_dag @ optimal_v + N @ optimal_alpha
            if (np.abs(_dq) > dq_max).any():
                _dq[np.abs(_dq) < 1e-5] = 1e-5
                beta = np.min(dq_max / np.abs(_dq))
                optimal_dq = _dq * beta
                print('beta_scale')
            else:
                optimal_dq = _dq
        else:
            print('not success')
            qdf = np.linalg.lstsq(jacb_star, v, rcond=None)[0]
            max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
            optimal_dq = max_gain * qdf
        v_optimal = jacb_star @ optimal_dq

        v_optimal = np.linalg.norm(v_optimal[:2])
        v_gain = np.linalg.norm(v_gain[:2])
        optima_gain_err = v_optimal - v_gain
        self.err.append(optima_gain_err)

    def with_without_manipulator_err(self, hit_dir, hit_pos):
        v = hit_dir
        success_inv, initial_q = inverse_kinematics(robot_model, robot_data, hit_pos)
        dq_min = -0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
        dq_max = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])

        def FKp(q):
            return forward_kinematics(robot_model, robot_data, q)[0]

        def hitting_objective(q, v):
            Jp = jacobian(robot_model, robot_data, q)[:3]
            manipulability = np.linalg.norm(np.dot(v, Jp))
            return -manipulability

        def constraint_ineq(q):
            return dq_max - np.abs(q)

        constraints = [{'type': 'eq', 'fun': lambda q: FKp(q) - hit_pos},
                       {'type': 'ineq', 'fun': constraint_ineq}]
        result_q = minimize(hitting_objective, initial_q, args=(v,), constraints=constraints)
        if not result_q.success:
            optimal_q = result_q.x
        else:
            optimal_q = initial_q

        v_optimal = self.compute_v(v=v, q=optimal_q, dq_max=dq_max, dq_min=dq_min)
        v_initial = self.compute_v(v=v, q=initial_q, dq_max=dq_max, dq_min=dq_min)

        v_optimal = np.linalg.norm(v_optimal[:2])
        v_initial = np.linalg.norm(v_initial[:2])
        optima_gain_err = v_optimal - v_initial
        self.err.append(optima_gain_err)

    def plot_histogram(self):
        hit_dirs, hit_poses = self.sample_pos_and_direction()
        for i in range(len(hit_dirs)):
            # self.optimal_gain_err(np.array([*hit_dirs[i], 0.]), np.array([*hit_poses[i], DESIRED_HEIGHT]))
            self.with_without_manipulator_err(np.array([*hit_dirs[i], 0.]), np.array([*hit_poses[i], DESIRED_HEIGHT]))
        bins = [-np.inf, -1, -0.5, -0.1, 0, 0.001, 0.1, 0.5, 1, np.inf]
        counts, _ = np.histogram(self.err, bins=bins)

        fig, ax = plt.subplots()
        bars = ax.bar(range(len(counts)), counts, tick_label=[f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)])
        ax.set_xlabel('range')
        ax.set_ylabel('number')
        ax.set_title(f'optimal-initial')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')
        plt.savefig('optimal_gain_err.PNG')
        plt.show()

    def compute_v(self, v, q, dq_max, dq_min):
        jacb_star = jacobian(robot_model, robot_data, q)[:3]

        vv = v.T.dot(v)
        c = [-vv, 0, 0, 0, 0, 0, 0, 0]
        J_dag = np.linalg.pinv(jacb_star)
        _JJ = J_dag.dot(jacb_star)
        N = np.eye(*_JJ.shape) - _JJ

        J_dag_v = J_dag
        J_dag_v = J_dag_v @ v

        A = np.c_[J_dag_v, N]
        A = np.r_[A, -A]
        _dq_max = 0.999 * dq_max
        _dq_min = 0.999 * dq_min
        _ub = np.c_[[*_dq_max, *(-_dq_min)]]
        result_eta = linprog(c=c, A_ub=A, b_ub=_ub, method='highs', x0=[0, 0, 0, 0, 0, 0, 0, 0], bounds=((0, None),
                             (dq_min[0],dq_max[0]), (dq_min[1],dq_max[1]), (dq_min[2],dq_max[2]), (dq_min[3],dq_max[3]),
                             (dq_min[4],dq_max[4]), (dq_min[5],dq_max[5]), (dq_min[6],dq_max[6])))
        if result_eta.success:
            optimal_eta = result_eta.x[0]
            optimal_alpha = result_eta.x[1:]
            optimal_v = optimal_eta * v
            _dq = J_dag @ optimal_v + N @ optimal_alpha
            if (np.abs(_dq) > dq_max).any():
                _dq[np.abs(_dq) < 1e-5] = 1e-5
                beta = np.min(dq_max / np.abs(_dq))
                optimal_dq = _dq * beta
                print('beta_scale')
            else:
                optimal_dq = _dq
        else:
            print('not success')
            qdf = np.linalg.lstsq(jacb_star, v, rcond=None)[0]
            max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
            optimal_dq = max_gain * qdf
        v_optimal = jacb_star @ optimal_dq
        return v_optimal


if __name__ == '__main__':
    plot_loss = PlotDataLoss()
    plot_loss.plot_loss()
    # plot_err = CompareMaxVel()
    # plot_err.plot_histogram()
