import numpy as np
import torch
from copy import deepcopy
from scipy.interpolate import interp1d
from hrl_air_hockey.bspline_planner.utils.bspline import BSpline
from .utils.constants import TableConstraint
from .utils.compute_vel_and_pos import compute_vel_and_pos


class TrajectoryPlanner:
    def __init__(self, planner_path, env_info, config, device):
        self.model = torch.load(planner_path)
        self.air_hockey_dt = env_info['dt']
        self.robot_model = deepcopy(env_info['robot']['robot_model'])
        self.robot_data = deepcopy(env_info['robot']['robot_data'])
        self.desired_height = TableConstraint.Z
        self.b_spline_q = BSpline(num_pts=config.bspline_q.n_ctr_pts, degree=config.bspline_q.degree,
                                  num_T_pts=config.bspline_q.num_T_pts, device=device)
        self.b_spline_t = BSpline(num_pts=config.bspline_t.n_ctr_pts, degree=config.bspline_t.degree,
                                  num_T_pts=config.bspline_t.num_T_pts, device=device)

    def plan_trajectory(self, q_0, dq_0, hit_pos, hit_dir, hit_scale):
        ddq_0 = np.zeros_like(q_0)
        q_f, dq_f = compute_vel_and_pos(self.robot_model, self.robot_data, np.array([*hit_pos, self.desired_height]), np.array([*hit_dir, 0.]),
                                        scale=hit_scale, initial_q=q_0)
        ddq_f = np.zeros_like(q_0)

        with torch.no_grad():
            features = torch.as_tensor(np.concatenate([q_0, dq_0, ddq_0, q_f, dq_f, ddq_f]))[None, :]
            q_cps, t_cps = self.model(features.to(torch.float32))
            q_cps, t_cps = q_cps.to(torch.float32), t_cps.to(torch.float32)
        return self.interpolate_control_points(q_cps, t_cps)

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
