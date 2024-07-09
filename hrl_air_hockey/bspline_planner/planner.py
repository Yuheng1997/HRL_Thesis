import numpy as np
import torch
import csv
from copy import deepcopy
from scipy.interpolate import interp1d
from hrl_air_hockey.bspline_planner.utils.bspline import BSpline
from .utils.constants import TableConstraint
from .utils.compute_vel_and_pos import compute_vel_and_pos
from air_hockey_challenge.utils.kinematics import jacobian, forward_kinematics


class TrajectoryPlanner:
    def __init__(self, planner_path, env_info, config, device, violate_path):
        self.model = torch.load(planner_path, map_location=torch.device(device))
        self.env_info = env_info
        self.air_hockey_dt = env_info['dt']
        self.robot_model = deepcopy(env_info['robot']['robot_model'])
        self.robot_data = deepcopy(env_info['robot']['robot_data'])
        self.desired_height = TableConstraint.Z
        self.b_spline_q = BSpline(num_pts=config.bspline_q.n_ctr_pts, degree=config.bspline_q.degree,
                                  num_T_pts=config.bspline_q.num_T_pts, device=device)
        self.b_spline_t = BSpline(num_pts=config.bspline_t.n_ctr_pts, degree=config.bspline_t.degree,
                                  num_T_pts=config.bspline_t.num_T_pts, device=device)
        self.huber = torch.nn.HuberLoss(reduction='none')
        self.violate_data_path = violate_path
        self.num_violate_point = 0

    def plan_trajectory(self, q_0, dq_0, hit_pos, hit_dir, hit_scale):
        ddq_0 = np.zeros_like(q_0)
        q_f, dq_f = compute_vel_and_pos(self.robot_model, self.robot_data, np.array([*hit_pos, self.desired_height]),
                                        np.array([*hit_dir, 0.]),
                                        scale=hit_scale, initial_q=q_0)
        ddq_f = np.zeros_like(q_0)

        with torch.no_grad():
            features = torch.as_tensor(np.concatenate([q_0, dq_0, ddq_0, q_f, dq_f, ddq_f]))[None, :]
            q_cps, t_cps = self.model(features.to(torch.float32))
            q_cps, t_cps = q_cps.to(torch.float32), t_cps.to(torch.float32)
        traj = self.interpolate_control_points(q_cps, t_cps)

        # check traj:
        good_traj = self.check_traj_violation(traj=traj[0])
        if not good_traj:
            self.add_violate_traj(q_0=q_0, dq_0=dq_0, ddq_0=ddq_0, q_f=q_f, dq_f=dq_f, ddq_f=ddq_f, pos_2d_end=hit_pos[:2])

        return traj

    def interpolate_control_points(self, q_cps, t_cps):
        with torch.no_grad():
            q = torch.einsum('ijk,lkm->ljm', self.b_spline_q.N, q_cps)
            q_dot_tau = torch.einsum('ijk,lkm->ljm', self.b_spline_q.dN, q_cps)
            q_ddot_tau = torch.einsum('ijk,lkm->ljm', self.b_spline_q.ddN, q_cps)

            dtau_dt = torch.einsum('ijk,lkm->ljm', self.b_spline_t.N, t_cps)
            ddtau_dtt = torch.einsum('ijk,lkm->ljm', self.b_spline_t.dN, t_cps)

            dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
            t_cumsum = torch.cumsum(dt, dim=-1)

            q_dot = q_dot_tau * dtau_dt
            q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

            traj = list()

            t_cumsum = t_cumsum.numpy()
            q = q.numpy()
            q_dot = q_dot.numpy()
            q_ddot = q_ddot.numpy()
            for n in range(q_cps.shape[0]):
                _dt = t_cumsum[n]
                _q = q[n]
                _q_dot = q_dot[n]
                _q_ddot = q_ddot[n]

                q_interpol = [interp1d(_dt, _q[:, i], kind='linear', fill_value='extrapolate') for i in range(7)]
                q_dot_interpol = [interp1d(_dt, _q_dot[:, i], kind='linear', fill_value='extrapolate') for i in
                                  range(7)]
                q_ddot_interpol = [interp1d(_dt, _q_ddot[:, i], kind='linear', fill_value='extrapolate') for i in
                                   range(7)]

                # ts = np.arange(0, dt[-1], air_hockey_dt / 20)
                _end_t = t_cumsum[n, -1]
                _start_t = self.air_hockey_dt

                ts = np.arange(start=_start_t, stop=_end_t, step=self.air_hockey_dt)

                _pos = np.array([q_interpol[i](ts) for i in range(7)]).transpose()
                _vel = np.array([q_dot_interpol[i](ts) for i in range(7)]).transpose()
                _acc = np.array([q_ddot_interpol[i](ts) for i in range(7)]).transpose()
                traj.append(np.concatenate([_pos, _vel, _acc], axis=-1))
        return traj

    def generate_return_traj(self, q_0, dq_0):
        # inverse kinematics
        q, qd = q_0, dq_0
        ee_pos = forward_kinematics(mj_model=self.robot_model, mj_data=self.robot_data, q=q_0)[0][:3]
        x_home = np.array([0.65, 0., self.env_info['robot']['ee_desired_height']])
        qd_max = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
        J = jacobian(mj_model=self.robot_model, mj_data=self.robot_data, q=q)[:3]
        v_des = x_home - ee_pos
        J_p = np.linalg.pinv(J)
        JJ = J_p.dot(J)
        I_n = np.eye(JJ.shape[0])
        qd_0 = (np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.]) - q) * 0.5
        qd_des = J_p.dot(v_des) + (I_n - JJ).dot(qd_0)
        gain = np.min(qd_max / np.abs(qd_des))
        qd_des = qd_des * gain
        q_des = q + qd_des * 0.02
        return [[*q_des, *qd_des]]

    def check_traj_violation(self, traj):
        # check constraint, obstacle
        positions = []
        for i in range(len(traj)):
            position = forward_kinematics(mj_model=self.robot_model, mj_data=self.robot_data, q=traj[i][:7])[0][:3]
            positions.append(position)
        constraint_loss, x_loss, y_loss, z_loss = self.constraint_loss(positions, 0.01)
        # print('constraint_loss', constraint_loss)
        if constraint_loss > 0.1:
            self.num_violate_point += 1
            return False
        else:
            return True

    def add_violate_traj(self, q_0, dq_0, ddq_0, q_f, dq_f, ddq_f, pos_2d_end):
        pos_2d_start = forward_kinematics(mj_model=self.robot_model, mj_data=self.robot_data, q=q_0)[0][:2]
        with open(self.violate_data_path, 'a', newline='') as file:
            writer = csv.writer(file)
            data_1 = np.concatenate([q_0, dq_0, ddq_0, pos_2d_start])
            data_2 = np.concatenate([q_f, dq_f, ddq_f, pos_2d_end])
            writer.writerow(data_1.tolist())
            writer.writerow(data_2.tolist())

    def constraint_loss(self, position, dt):
        position_array = np.array(position)
        ee_pos = torch.tensor(position_array)
        huber_along_path = lambda x: dt * self.huber(x, torch.zeros_like(x))
        relu_huber_along_path = lambda x: huber_along_path(torch.relu(x))
        x_b = torch.tensor([0.6, 1.31])
        y_b = torch.tensor([-0.47085, 0.47085])
        z = torch.tensor(0.1645)
        x_loss = relu_huber_along_path(x_b[0] - ee_pos[:, 0]) + relu_huber_along_path(ee_pos[:, 0] - x_b[1])
        y_loss = relu_huber_along_path(y_b[0] - ee_pos[:, 1]) + relu_huber_along_path(ee_pos[:, 1] - y_b[1])
        z_loss = relu_huber_along_path(z - ee_pos[:, 2]) + relu_huber_along_path(ee_pos[:, 2] - z)
        constraint_losses = torch.sum(x_loss + y_loss + z_loss)
        return constraint_losses, x_loss, y_loss, z_loss
