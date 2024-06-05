import os
import torch
import numpy as np
from config import Config
from utils.bspline import BSpline
from scipy.interpolate import interp1d
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper


device = 'cpu'
air_hockey_dt = 0.02
b_spline_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                    num_T_pts=Config.bspline_q.num_T_pts, device=device)
b_spline_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                    num_T_pts=Config.bspline_t.num_T_pts, device=device)


def load_data(filename, n=100):
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
    env._write_data("puck_x_pos", x)
    env._write_data("puck_y_pos", y)
    env._write_data("puck_x_vel", 0)
    env._write_data("puck_y_vel", 0)
    env._write_data("puck_yaw_vel", 0)
    # hit_range = np.array([[0.8, 1.3], [-0.39105, 0.39105]])
    # puck_pos = np.random.rand(2) * (hit_range[:, 1] - hit_range[:, 0]) + hit_range[:, 0]


def compute_control_points(model, features):
    with torch.no_grad():
        features_tensor = features.clone().detach()
        q_cps, t_cps = model(features_tensor)
        q_cps, t_cps = q_cps.to(torch.float32), t_cps.to(torch.float32)
    return q_cps, t_cps


def main():
    # replay model and date
    n_sample = 100
    # compute trajectory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    feature_path = os.path.abspath(os.path.join(current_dir, os.pardir, 'low_level/datasets/uniform_train/data.tsv'))
    features = load_data(filename=feature_path, n=n_sample)

    model_path = os.path.abspath(os.path.join(current_dir, os.pardir, 'trained_low_agent/Model_2020.pt'))
    # model_path = os.path.join('Model_20000_new.pt')
    model = torch.load(model_path, map_location=torch.device(device))

    q_c_ps, t_c_ps = compute_control_points(model, features[:, :42])
    pos_traj, vel_traj = interpolate_control_points(q_c_ps, t_c_ps)

    # render
    from hit_back_env import HitBackEnv
    # env = AirHockeyChallengeWrapper(env="7dof-hit", interpolation_order=3, debug=True)
    env = HitBackEnv(horizon=1000)
    env.reset()
    for i in range(n_sample):
        initial_obs = np.zeros(46)
        joint_pos = features[i, :7]
        joint_vel = features[i, 7:14]
        initial_obs[6:13] = joint_pos
        initial_obs[13:20] = joint_vel
        env.reset(initial_obs)
        set_puck_pos(env, features[i, 42:])
        print('number of exp:', i)
        for j in range(pos_traj[i].shape[0]):
            action = np.vstack([pos_traj[i][j], vel_traj[i][j]])
            action = np.array([action, np.zeros((2, 7))])
            env.step(action)
            env.render()


if __name__ == '__main__':
    main()