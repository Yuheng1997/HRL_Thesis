import os
import torch
import analysis
import numpy as np
from config import Config
from utils.bspline import BSpline
from scipy.interpolate import interp1d
from generate_hitting_data import get_hitting_sample, generate_data_smash
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper


device = 'cpu'
air_hockey_dt = 0.02
b_spline_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                    num_T_pts=Config.bspline_q.num_T_pts, device=device)
b_spline_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                    num_T_pts=Config.bspline_t.num_T_pts, device=device)
n_sample = 100


def load_data(filename="test_data_challenge_defend.tsv", n=n_sample):
    dataset_path = filename
    # dataset_path = dataset_path.replace("train", "test")
    # dataset_path = dataset_path.replace("data.tsv", filename)
    data = np.loadtxt(dataset_path, delimiter='\t').astype(np.float32)

    # np.random.shuffle(data)
    data = data[:n]
    features = torch.from_numpy(data).to(device)

    return features


def interpolate_control_points(q_cps, t_cps):
    with torch.no_grad():
        q = torch.einsum('ijk,lkm->ljm', b_spline_q.N, q_cps)
        q_dot_tau = torch.einsum('ijk,lkm->ljm', b_spline_q.dN, q_cps)
        q_ddot_tau = torch.einsum('ijk,lkm->ljm', b_spline_q.ddN, q_cps)

        dtau_dt = torch.einsum('ijk,lkm->ljm', b_spline_t.N, t_cps)
        ddtau_dtt = torch.einsum('ijk,lkm->ljm', b_spline_t.dN, t_cps)

        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
        t_cumsum = torch.cumsum(dt, dim=-1)

        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

        pos = list()
        vel = list()
        acc = list()

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
            q_dot_interpol = [interp1d(_dt, _q_dot[:, i], kind='linear', fill_value='extrapolate') for i in range(7)]
            q_ddot_interpol = [interp1d(_dt, _q_ddot[:, i], kind='linear', fill_value='extrapolate') for i in range(7)]

            # ts = np.arange(0, dt[-1], air_hockey_dt / 20)
            _end_t = t_cumsum[n, -1]
            _start_t = air_hockey_dt

            ts = np.arange(start=_start_t, stop=_end_t, step=air_hockey_dt)

            _pos = np.array([q_interpol[i](ts) for i in range(7)]).transpose()
            _vel = np.array([q_dot_interpol[i](ts) for i in range(7)]).transpose()
            _acc = np.array([q_ddot_interpol[i](ts) for i in range(7)]).transpose()
            pos.append(_pos)
            vel.append(_vel)
            acc.append(_acc)

    return pos, vel


def set_puck_pos(env, desierd_puck_pos):
    x, y = desierd_puck_pos - torch.tensor([1.51, 0])
    env.base_env._write_data("puck_x_pos", x)
    env.base_env._write_data("puck_y_pos", y)
    env.base_env._write_data("puck_x_vel", 0)
    env.base_env._write_data("puck_y_vel", 0)
    env.base_env._write_data("puck_yaw_vel", 0)
    # hit_range = np.array([[0.8, 1.3], [-0.39105, 0.39105]])
    # puck_pos = np.random.rand(2) * (hit_range[:, 1] - hit_range[:, 0]) + hit_range[:, 0]


def main():
    # replay model and date
    set_nummber = False
    number = 6
    # compute trajectory
    features = load_data(filename='datasets/uniform_train/data.tsv')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, os.pardir, 'trained_low_agent/in1_3table.pt'))
    # model_path = os.path.join('Model_20000_new.pt')
    model = torch.load(model_path, map_location=torch.device(device))

    q_c_ps, t_c_ps = analysis.compute_control_points(model, features[:, :42])
    pos_traj, vel_traj = interpolate_control_points(q_c_ps, t_c_ps)

    # render
    env = AirHockeyChallengeWrapper(env="7dof-hit", interpolation_order=3, debug=True)
    env.reset()
    for i in range(n_sample):
        if not set_nummber:
            if i % 2 == 0:
                env.reset()
                set_puck_pos(env, features[i, 42:])
                print('number of exp:', i)
            for j in range(pos_traj[i].shape[0]):
                action = np.vstack([pos_traj[i][j], vel_traj[i][j]])
                env.step(action)
                env.render()
        else:
            env.reset()
            set_puck_pos(env, features[number, 42:])
            print('number of exp:', number)
            for j in range(pos_traj[number].shape[0]):
                action = np.vstack([pos_traj[number][j], vel_traj[number][j]])
                env.step(action)
                env.render()
            for j in range(pos_traj[number+1].shape[0]):
                action = np.vstack([pos_traj[number+1][j], vel_traj[number+1][j]])
                env.step(action)
                env.render()


if __name__ == '__main__':
    main()