import numpy as np
from scipy.optimize import linprog
from .kinematics import inverse_kinematics, jacobian
from .constants import Limits


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
    result_eta = linprog(c=c, A_ub=A, b_ub=_ub, method='highs', x0=[0, 0, 0, 0, 0, 0, 0, 0], bounds=bounds)
    if result_eta.success:
        optimal_eta = result_eta.x[0]
        optimal_alpha = result_eta.x[1:]
        optimal_v = optimal_eta * v
        _dq = (J_dag @ optimal_v + N @ optimal_alpha) * scale
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