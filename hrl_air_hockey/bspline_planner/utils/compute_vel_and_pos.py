import numpy as np
from .constants import Limits
from scipy.optimize import linprog
from .kinematics import inverse_kinematics, jacobian
from hrl_air_hockey.bspline_planner.utils.constants import RobotEnvInfo


table_width = RobotEnvInfo.table_width
table_length = RobotEnvInfo.table_length
puck_radius = RobotEnvInfo.puck_radius
mallet_radius = RobotEnvInfo.mallet_radius


def compute_vel_and_pos(robot_model, robot_data, hitting_point, hitting_direction, scale, initial_q=None):
    v = hitting_direction
    success_inv, initial_q = inverse_kinematics(robot_model, robot_data, hitting_point,
                                                initial_q=initial_q)
    dq_min = -Limits.q_dot7.cpu().detach().numpy()
    dq_max = Limits.q_dot7.cpu().detach().numpy()

    jacb_star = jacobian(robot_model, robot_data, initial_q)[:3]
    # min{c.T @ x}
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
    bounds = np.zeros((8, 2))
    bounds[1:, 0] = dq_min
    bounds[1:, 1] = dq_max
    bounds[0, 1] = 10
    result_eta = linprog(c=c, A_ub=A, b_ub=_ub, method='highs', bounds=bounds)
    if result_eta.success:
        optimal_eta = result_eta.x[0]
        optimal_alpha = result_eta.x[1:]
        optimal_v = optimal_eta * v
        clipped_v = clip_v(optimal_v, hitting_point, v)
        _dq = (J_dag @ clipped_v + N @ optimal_alpha) * scale
        if (np.abs(_dq) > dq_max).any():
            _dq[np.abs(_dq) < 1e-5] = 1e-5
            beta = np.min(dq_max / np.abs(_dq))
            optimal_dq = _dq * beta
        else:
            optimal_dq = _dq
    else:
        qdf = np.linalg.lstsq(jacb_star, v, rcond=None)[0]
        max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
        optimal_dq = max_gain * qdf * scale
    return initial_q, optimal_dq


def clip_v(v, hitting_point, hitting_dir):
    v_x = clip_eta_x(v[0], hitting_point[0], hitting_dir[0])
    v_y = clip_eta_y(v[1], hitting_point[1], hitting_dir[1])
    clipped_v = np.array([v_x, v_y, v[2]])
    return clipped_v


def clip_eta_y(eta_y, point_pos_y, point_vel_y):
    dis = (mallet_radius + puck_radius) * 2
    point_dis = (table_width / 2 - mallet_radius) - np.abs(point_pos_y)
    ratio = point_dis / dis
    if ratio < 1:
        if point_pos_y > 0 and point_vel_y > 0:
            return np.clip(a=eta_y, a_min=0, a_max=ratio * 0.5)
        if point_pos_y < 0 and point_vel_y < 0:
            return np.clip(a=eta_y, a_min=0, a_max=ratio * 0.5)
        return eta_y
    else:
        return eta_y


def clip_eta_x(eta_x, point_pos_x, point_vel_x):
    dis = (mallet_radius + puck_radius) * 2
    point_dis = point_pos_x - 0.60
    ratio = point_dis / dis
    if ratio < 1:
        if point_vel_x < 0:
            return np.clip(a=eta_x, a_min=0, a_max=ratio * 0.5)
        else:
            return eta_x
    else:
        return eta_x
