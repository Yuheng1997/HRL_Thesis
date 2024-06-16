import os
import numpy as np
import torch
import time
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, linprog
from config import Config
from utils.bspline import BSpline
from utils.constants import Limits
from utils.constants import TableConstraint


DESIRED_HEIGHT = TableConstraint.Z
mallet_radius = 0.04815
puck_radius = 0.03165
table_width = 1.038
table_length = 1.948
goal_width = 0.25
robot_offset = np.array([-1.51, 0, -0.1])

PUCK_DEFAULT_POS = [-100, -100]

robot_model = mujoco.MjModel.from_xml_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/iiwa_only.xml"))
robot_model.body('iiwa_1/base').pos = np.zeros(3)
robot_data = mujoco.MjData(robot_model)

## ===============================================================================
##                                  General functions
## ===============================================================================


def _mujoco_clik(desired_pos, desired_quat, initial_q, name, model, data, lower_limit, upper_limit):
    IT_MAX = 1000
    eps = 1e-4
    damp = 1e-3
    progress_thresh = 20.0
    max_update_norm = 0.1
    rot_weight = 1
    i = 0

    dtype = data.qpos.dtype

    data.qpos = initial_q

    neg_x_quat = np.empty(4, dtype=dtype)
    error_x_quat = np.empty(4, dtype=dtype)

    if desired_pos is not None and desired_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if desired_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif desired_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError("Desired Position and desired rotation is None, cannot compute inverse kinematics")

    while True:
        # forward kinematics
        mujoco.mj_fwdPosition(model, data)

        x_pos = data.body(name).xpos
        x_quat = data.body(name).xquat

        error_norm = 0
        if desired_pos is not None:
            err_pos[:] = desired_pos - x_pos
            error_norm += np.linalg.norm(err_pos)

        if desired_quat is not None:
            mujoco.mju_negQuat(neg_x_quat, x_quat)
            mujoco.mju_mulQuat(error_x_quat, desired_quat, neg_x_quat)
            mujoco.mju_quat2Vel(err_rot, error_x_quat, 1)
            error_norm += np.linalg.norm(err_rot) * rot_weight

        if error_norm < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break

        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)

        hess_approx = jac.T.dot(jac)
        joint_delta = jac.T.dot(err)

        hess_approx += np.eye(hess_approx.shape[0]) * damp
        update_joints = np.linalg.solve(hess_approx, joint_delta)

        update_norm = np.linalg.norm(update_joints)

        # Check whether we are still making enough progress, and halt if not.
        progress_criterion = error_norm / update_norm
        if progress_criterion > progress_thresh:
            success = False
            break

        if update_norm > max_update_norm:
            update_joints *= max_update_norm / update_norm

        mujoco.mj_integratePos(model, data.qpos, update_joints, 1)
        data.qpos = np.clip(data.qpos, lower_limit, upper_limit)
        i += 1
    q_cur = data.qpos.copy()

    return success, q_cur
        

def link_to_xml_name(mj_model, link):
    try:
        mj_model.body('iiwa_1/base')
        link_to_frame_idx = {
            "1": "iiwa_1/link_1",
            "2": "iiwa_1/link_2",
            "3": "iiwa_1/link_3",
            "4": "iiwa_1/link_4",
            "5": "iiwa_1/link_5",
            "6": "iiwa_1/link_6",
            "7": "iiwa_1/link_7",
            "ee": "iiwa_1/striker_joint_link",
        }
    except:
        link_to_frame_idx = {
            "1": "planar_robot_1/body_1",
            "2": "planar_robot_1/body_2",
            "3": "planar_robot_1/body_3",
            "ee": "planar_robot_1/body_ee",
        }
    return link_to_frame_idx[link]


def forward_kinematics(mj_model, mj_data, q, link="ee"):
    """
    Compute the forward kinematics of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is the
        case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        q (np.array):
            joint configuration for which the forward kinematics are computed
        link (string, "ee"):
            Link for which the forward kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"]. When using planar the choices are ["1", "2", "3", "ee"]

    Returns
    -------
    position: numpy.ndarray, (3,)
        Position of the link in robot's base frame
    orientation: numpy.ndarray, (3, 3)
        Orientation of the link in robot's base frame
    """

    return _mujoco_fk(q, link_to_xml_name(mj_model, link), mj_model, mj_data)


def _mujoco_fk(q, name, model, data):
    data.qpos[:len(q)] = q
    mujoco.mj_fwdPosition(model, data)
    return data.body(name).xpos.copy(), data.body(name).xmat.reshape(3, 3).copy()


def inverse_kinematics(mj_model, mj_data, desired_position, desired_rotation=None, initial_q=None, link="ee"):
    q_init = np.zeros(mj_model.nq)
    if initial_q is None:
        q_init = mj_data.qpos
    else:
        q_init[:initial_q.size] = initial_q

    q_l = mj_model.jnt_range[:, 0]
    q_h = mj_model.jnt_range[:, 1]
    lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
    upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

    desired_quat = None
    if desired_rotation is not None:
        desired_quat = np.zeros(4)
        mujoco.mju_mat2Quat(desired_quat, desired_rotation.reshape(-1, 1))

    return _mujoco_clik(desired_position, desired_quat, q_init, link_to_xml_name(mj_model, link), mj_model,
                        mj_data, lower_limit, upper_limit)


def solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, q_0):
        hit_pos = np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]])
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        success, q_star = solve_hit_config_ik_null(hit_pos, hit_dir, q_0)
        if not success:
            print("Failed!")
            q_star = q_0
        return q_star, success


def solve_hit_config_ik_null(x_des, v_des, q_0, max_time=100):
        t_start = time.time()
        reg = 0e-6
        dim = q_0.shape[0]
        IT_MAX = 1000
        eps = 1e-4
        damp = 1e-3
        progress_thresh = 20.0
        max_update_norm = 0.1
        i = 0
        TIME_MAX = max_time
        success = False

        dtype = np.float64

        robot_data.qpos = q_0

        q_l = robot_model.jnt_range[:, 0]
        q_h = robot_model.jnt_range[:, 1]
        lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
        upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

        name = link_to_xml_name(robot_model, 'ee')

        def objective(q, grad):
            if grad.size > 0:
                grad[...] = numerical_grad(objective, q)
            f = v_des @ jacobian(robot_model, robot_data, q)[:3, :dim]
            return f @ f + reg * np.linalg.norm(q - q_0)

        null_opt_stop_criterion = False
        while True:
            # forward kinematics
            mujoco.mj_fwdPosition(robot_model, robot_data)

            x_pos = robot_data.body(name).xpos

            err_pos = x_des - x_pos
            error_norm = np.linalg.norm(err_pos)

            f_grad = numerical_grad(objective, robot_data.qpos.copy())
            f_grad_norm = np.linalg.norm(f_grad)
            if f_grad_norm > max_update_norm:
                f_grad = f_grad / f_grad_norm

            if error_norm < eps:
                success = True
            if time.time() - t_start > TIME_MAX or i >= IT_MAX or null_opt_stop_criterion:
                break

            jac_pos = np.empty((3, robot_model.nv), dtype=dtype)
            mujoco.mj_jacBody(robot_model, robot_data, jac_pos, None, robot_model.body(name).id)

            update_joints = jac_pos.T @ np.linalg.inv(jac_pos @ jac_pos.T + damp * np.eye(3)) @ err_pos

            # Add Null space Projection
            null_dq = (np.eye(robot_model.nv) - np.linalg.pinv(jac_pos) @ jac_pos) @ f_grad
            null_opt_stop_criterion = np.linalg.norm(null_dq) < 1e-4
            update_joints += null_dq

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = error_norm / update_norm
            if progress_criterion > progress_thresh:
                success = False
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            mujoco.mj_integratePos(robot_model, robot_data.qpos, update_joints, 1)
            robot_data.qpos = np.clip(robot_data.qpos, lower_limit, upper_limit)
            i += 1
        q_cur = robot_data.qpos.copy()

        return success, q_cur


def jacobian(mj_model, mj_data, q, link="ee"):
    return _mujoco_jac(q, link_to_xml_name(mj_model, link), mj_model, mj_data)


def _mujoco_jac(q, name, model, data):
    data.qpos[:len(q)] = q
    dtype = data.qpos.dtype
    jac = np.empty((6, model.nv), dtype=dtype)
    jac_pos, jac_rot = jac[:3], jac[3:]
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)
    return jac


def numerical_grad(fun, q):
    eps = np.sqrt(np.finfo(np.float64).eps)
    grad = np.zeros_like(q)
    for i in range(q.shape[0]):
        q_pos = q.copy()
        q_neg = q.copy()
        q_pos[i] += eps
        q_neg[i] -= eps
        grad[i] = (fun(q_pos, np.array([])) - fun(q_neg, np.array([]))) / 2 / eps
    return grad


def get_init_configuration():
    init_state = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
    init_pos = np.array([0.65, 0., DESIRED_HEIGHT]) + np.random.uniform(low=-0.05, high=0.05, size=(3,))
    init_pos[2] = DESIRED_HEIGHT + 0.005 * (2*np.random.rand() - 1.)

    success, init_state = inverse_kinematics(robot_model,
                                             robot_data,
                                             init_pos,
                                             R.from_euler('xyz', [0, 5 / 6 * np.pi, 0]).as_matrix(),
                                             initial_q=init_state)
    
    assert success is True
    return init_state, init_pos


def get_initial_qd(var=0.25):
    return np.zeros((7,)) + np.random.uniform(low=-var, high=var, size=(7,))


def get_initial_qd_new(var=0.15):
    return np.random.normal(loc=0., scale=var, size=(7,))


def get_hitting_configuration(q0, hit_pos_2d, hit_dir_2d):
        success, qf = inverse_kinematics(robot_model, robot_data, 
                                    np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]), desired_rotation=None, initial_q=q0, link="ee")
        
        qf_optimal, success = solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, qf)
        if success: 
            return qf_optimal
        else:
            qf_optimal, success = solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, np.zeros((7,)))
            if success:
                return qf_optimal
            
        return qf


def get_qd_max(hit_dir_2d, qf):
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        jac = jacobian(robot_model, robot_data, qf)[:3]
        qdf = np.linalg.lstsq(jac, hit_dir, rcond=None)[0]
        max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
        qdf *= max_gain

        # print(jac @ qdf)
        # print(np.linalg.norm(jac @ qdf))
        
        return qdf


def get_qd_max_3d(hit_dir, qf):
        jac = jacobian(robot_model, robot_data, qf)[:3]
        qdf = np.linalg.lstsq(jac, hit_dir, rcond=None)[0]
        max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
        qdf *= max_gain

        # print(jac[:2] @ qdf)
        print('v', jac @ qdf)
        # print(np.linalg.norm(jac[:2] @ qdf))

        return qdf, np.linalg.norm(jac[:2] @ qdf)


## ===============================================================================
##                                  Prepare
## ===============================================================================


def get_init_puck_pos_prepare():
    if np.random.rand() < 0.5:  # Create puck at bottom range of table
        puck_pos = np.random.uniform(low=[0.57, -0.48535], high=[0.8, 0.48535])
    else:  # Create puck at side of table
        puck_pos = np.random.uniform(low=[0.57, 0.39105], high=[1.3, 0.48535])
        puck_pos *= [1, [1, -1][np.random.randint(2)]]

    return puck_pos


def get_prepare_sample(puck_pos_2d):
    prepare_range = [0.8, 1.3]

    if puck_pos_2d[0] < prepare_range[0]:
        hit_dir_2d = np.array([-1, np.sign(puck_pos_2d[1] + 1e-6) * 0.2])
        hit_vel_mag = 0.2
    elif abs(puck_pos_2d[0]) > np.mean(prepare_range):
        hit_dir_2d = np.array([-0.5, np.sign(puck_pos_2d[1] + 1e-6)])
        hit_vel_mag = 0.2
    else:
        hit_dir_2d = np.array([0, np.sign(puck_pos_2d[1] + 1e-6)])
        hit_vel_mag = 0.2

    hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
    hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (mallet_radius +
                                                     puck_radius)
    
    q0, init_pos = get_init_configuration()
    if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
        # print("Distance too short!")
        raise RuntimeError("Distance too short!")
    qd0 = get_initial_qd()
    qdd0 = np.zeros_like(q0)
            
    success, qf = inverse_kinematics(robot_model, robot_data, np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]), desired_rotation=None, initial_q=q0, link="ee")
    if not success:
        # print("!")
        raise RuntimeError("Distance too short!")
    if not np.all(np.isclose(forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))):
        # print("!!", forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))
        raise RuntimeError("Distance too short!")

    qdf = get_qd_max(hit_dir_2d, qf) * hit_vel_mag
    qddf = np.zeros_like(qf)
    
    s1 = np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, puck_pos_2d], axis=None)
    s2 = np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None)

    return s1, s2


def generate_data_prepare(n=5000):
    data = []

    while len(data) < n:
        print(len(data))
        puck_pos_2d = get_init_puck_pos_prepare()
        prepare_range = [0.8, 1.3]

        if puck_pos_2d[0] < prepare_range[0]:
            hit_dir_2d = np.array([-1, np.sign(puck_pos_2d[1] + 1e-6) * 0.2])
            hit_vel_mag = 0.2
        elif abs(puck_pos_2d[0]) > np.mean(prepare_range):
            hit_dir_2d = np.array([-0.5, np.sign(puck_pos_2d[1] + 1e-6)])
            hit_vel_mag = 0.2
        else:
            hit_dir_2d = np.array([0, np.sign(puck_pos_2d[1] + 1e-6)])
            hit_vel_mag = 0.2

        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (mallet_radius +
                                                     puck_radius)

        q0, init_pos = get_init_configuration()
        if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
            print("Distance too short!")
            continue
        qd0 = get_initial_qd()
        qdd0 = np.zeros_like(q0)
            
        success, qf = inverse_kinematics(robot_model, robot_data, np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]), desired_rotation=None, initial_q=q0, link="ee")
        if not success:
            print("!")
            continue
        if not np.all(np.isclose(forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))):
            print("!!", forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))
            continue

        qdf = get_qd_max(hit_dir_2d, qf) * hit_vel_mag
        qddf = np.zeros_like(qf)

        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, PUCK_DEFAULT_POS], axis=None))
        data.append(np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None))

    np.savetxt(f"prepare_data_{n}.tsv", data, delimiter='\t', fmt="%.10f")


def get_init_puck_pos_prepare_new():
    width_high = table_width / 2 - puck_radius - 0.002
    width_low = table_width / 2 - puck_radius - 2 * mallet_radius

    side_range = np.array([[0.8, 1.3], [width_low, width_high]])
    bottom_range = np.array([[0.57, 0.8], [goal_width / 2, width_high]])

    side_area = (side_range[0, 1] - side_range[0, 0]) * (side_range[1, 1] - side_range[1, 0])
    bottom_area = (bottom_range[0, 1] - bottom_range[0, 0]) * (bottom_range[1, 1] - bottom_range[1, 0])

    if np.random.rand() >= side_area / (side_area + bottom_area):
        start_range = bottom_range
    else:
        start_range = side_range

    puck_pos = np.random.rand(2) * (start_range[:, 1] - start_range[:, 0]) + start_range[:, 0]
    puck_pos *= [1, [1, -1][np.random.randint(2)]]

    return puck_pos


def generate_data_prepare_new(n=5000):
    data = []

    prepare_range = [0.8, 1.3]
    hit_vel_mag = 0.2

    while len(data) < n:
        print(len(data))
        puck_pos_2d = get_init_puck_pos_prepare_new()

        if puck_pos_2d[0] < prepare_range[0]:  # puck in bottom range: hit towards corner
            hit_dir_2d = np.array([-1, np.sign(puck_pos_2d[1] + 1e-6) * 0.2])
        elif abs(puck_pos_2d[0]) > np.mean(prepare_range):  # puck in far side range: hit towards own goal to the edge
            hit_dir_2d = np.array([-0.5, np.sign(puck_pos_2d[1] + 1e-6)])
        else:  # puck in middle side range: hit straight towards side
            hit_dir_2d = np.array([0, np.sign(puck_pos_2d[1] + 1e-6)])

        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (mallet_radius + puck_radius)

        q0, init_pos = get_init_configuration()
        if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
            print("Distance too short!")
            continue
        qd0 = get_initial_qd()
        qdd0 = np.zeros_like(q0)
            
        success, qf = inverse_kinematics(robot_model, robot_data, np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]), desired_rotation=None, initial_q=q0, link="ee")
        if not success:
            print("!")
            continue
        if not np.all(np.isclose(forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))):
            print("!!", forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))
            continue

        qdf = get_qd_max(hit_dir_2d, qf) * hit_vel_mag
        qddf = np.zeros_like(qf)

        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, PUCK_DEFAULT_POS], axis=None))
        data.append(np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None))

    np.savetxt(f"prepare_data_{n}.tsv", data, delimiter='\t', fmt="%.10f")


## ===============================================================================
##                                  Defend
## ===============================================================================


def get_init_puck_pos_defend():
    defend_range = np.array([[0.8, 1.0], [-0.45, 0.45]])

    puck_pos = np.random.rand(2) * (defend_range[:, 1] - defend_range[:, 0]) + defend_range[:, 0]
    return puck_pos


def get_defend_sample(puck_pos_2d):
    x_correction = - np.sin(np.pi/4) * (mallet_radius + puck_radius)
    y_correction = - np.cos(np.pi/4) * (mallet_radius + puck_radius)

    hit_pos_2d = puck_pos_2d + np.array([x_correction, np.sign(puck_pos_2d[1]) * y_correction])

    q0, init_pos = get_init_configuration()
    if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
        # print("Distance too short!")
        raise RuntimeError("Distance too short!")
    qd0 = get_initial_qd()
    qdd0 = np.zeros_like(q0)
            
    success, qf = inverse_kinematics(robot_model, robot_data, np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]), desired_rotation=None, initial_q=q0, link="ee")
    if not success:
        # print("!")
        raise RuntimeError("Distance too short!")
    if not np.all(np.isclose(forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))):
        # print("!!", forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))
        raise RuntimeError("Distance too short!")

    qdf = get_initial_qd()
    qddf = np.zeros_like(qf)
    
    s1 = np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, puck_pos_2d], axis=None)
    s2 = np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None)

    return s1, s2


def generate_data_defend(n=10000):
    data = []

    x_correction = - np.sin(np.pi/4) * (mallet_radius + puck_radius)
    y_correction = - np.cos(np.pi/4) * (mallet_radius + puck_radius)

    while len(data) < n:
        print(len(data))
        puck_pos_2d = get_init_puck_pos_defend()

        hit_pos_2d = puck_pos_2d + np.array([x_correction, np.sign(puck_pos_2d[1]) * y_correction])

        q0, init_pos = get_init_configuration()
        if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
            print("Distance too short!")
            continue
        qd0 = get_initial_qd()
        qdd0 = np.zeros_like(q0)
            
        success, qf = inverse_kinematics(robot_model, robot_data, np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]), desired_rotation=None, initial_q=q0, link="ee")
        if not success:
            print("!")
            continue
        if not np.all(np.isclose(forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))):
            print("!!", forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))
            continue

        qdf = get_initial_qd()
        qddf = np.zeros_like(qf)

        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, PUCK_DEFAULT_POS], axis=None))
        data.append(np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None))

    np.savetxt(f"defend_data_{n}.tsv", data, delimiter='\t', fmt="%.10f")


def get_init_puck_pos_defend_new():
    defend_range = np.array([[0.8, 1.0], [-0.45, 0.45]])

    puck_pos = np.random.rand(2) * (defend_range[:, 1] - defend_range[:, 0]) + defend_range[:, 0]
    return puck_pos


def generate_data_defend_new(n=10000):
    data = []
    hit_vel_mag = 0.05

    while len(data) < n:
        print(len(data))
        puck_pos_2d = get_init_puck_pos_defend_new()

        hit_dir_2d = np.array([0, np.sign(puck_pos_2d[1] + 1e-6)])
        hit_pos_2d = puck_pos_2d[:2] - (hit_dir_2d * mallet_radius)

        q0, init_pos = get_init_configuration()
        if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
            print("Distance too short!")
            continue
        qd0 = get_initial_qd()
        qdd0 = np.zeros_like(q0)
            
        success, qf = inverse_kinematics(robot_model, robot_data, np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]), desired_rotation=None, initial_q=q0, link="ee")
        if not success:
            print("!")
            continue
        if not np.all(np.isclose(forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))):
            print("!!", forward_kinematics(robot_model, robot_data, qf, link="ee")[0], np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]))
            continue

        qdf = get_qd_max(hit_dir_2d, qf) * hit_vel_mag
        qddf = np.zeros_like(qf)

        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, PUCK_DEFAULT_POS], axis=None))
        data.append(np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None))

    np.savetxt(f"defend_data_{n}.tsv", data, delimiter='\t', fmt="%.10f")


## ===============================================================================
##                                  Smash
## ===============================================================================


def get_init_puck_pos_smash():
    # hit_range = np.array([[0.8, 1.3], [-0.39105, 0.39105]])
    hit_range = np.array([[0.8, 1.3], [-0.37, -0.39105]])
    puck_pos = np.random.rand(2) * (hit_range[:, 1] - hit_range[:, 0]) + hit_range[:, 0]
    return puck_pos


def get_hitting_sample(puck_pos_2d):
    goal_pos = np.array([2.49, 0.0])

    hit_dir_2d = goal_pos - puck_pos_2d
    hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
    hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * mallet_radius

    q0, init_pos = get_init_configuration()
    if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
        # print("Distance too short!")
        raise RuntimeError("Distance too short!")
    qd0 = get_initial_qd()
    qdd0 = np.zeros_like(q0)

    # qf = get_hitting_configuration(q0, hit_pos_2d, hit_dir_2d)
    # qdf = get_qd_max(hit_dir_2d, qf)
    # qddf = np.zeros_like(qf)

    # print(qf)
    # print("Max_gain 1", max_gain)

    # Own optimize implementation:
    # =======================================================================================================
    qf = optimize(np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]), np.concatenate([hit_dir_2d, [0.]]))
    qdf = get_qd_max(hit_dir_2d, qf)
    qddf = np.zeros_like(qf)
    # print(qf)
    # print("Max_gain 2", max_gain)
    # ========================================================================================================
    
    s1 = np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, puck_pos_2d], axis=None)
    s2 = np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None)

    return s1, s2
    

def generate_data_smash(n=20000):
    data = []
    goal_pos = np.array([2.49, 0.0])

    while len(data) < n:
        print(len(data))
        puck_pos_2d = get_init_puck_pos_smash()

        hit_dir_2d = goal_pos - puck_pos_2d
        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * mallet_radius

        q0, init_pos = get_init_configuration()
        if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
            print("Distance too short!")
            continue
        qd0 = get_initial_qd()
        qdd0 = np.zeros_like(q0)

        # qf = get_hitting_configuration(q0, hit_pos_2d, hit_dir_2d)
        # qdf = get_qd_max(hit_dir_2d, qf)
        # qddf = np.zeros_like(qf)

        # print(qf)
        # print("Max_gain 1", max_gain)

        # Own optimize implementation:
        # =======================================================================================================
        qf = optimize(np.concatenate([hit_pos_2d, [DESIRED_HEIGHT]]), np.concatenate([hit_dir_2d, [0.]]))
        qdf = get_qd_max(hit_dir_2d, qf)
        qddf = np.zeros_like(qf)
        # print(qf)
        # print("Max_gain 2", max_gain)
        # ========================================================================================================

        qf2, direction = optimize_joint_angles_and_direction(hit_pos_2d, hit_dir_2d)
        qdf2 = get_qd_max_3d(direction, qf)
        qddf2 = np.zeros_like(qf)

        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, puck_pos_2d], axis=None))
        data.append(np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None))

    np.savetxt(f"hitting_data_{n}.tsv", data, delimiter='\t', fmt="%.10f")


def get_init_puck_pos_smash_new():
    y_lim = table_width / 2 - puck_radius - 2 * mallet_radius
    # hit_range = np.array([[0.8, 1.3], [-y_lim, y_lim]])
    hit_range = np.array([[0.8, 1.3], [-0.39105, 0.39105]])
    puck_pos = np.random.rand(2) * (hit_range[:, 1] - hit_range[:, 0]) + hit_range[:, 0]
    return puck_pos


def generate_data_smash_new(n=20000):
    data = []
    goal_pos = np.array([2.49, 0.0])
    c = {}

    while len(data) < n:
        # print(len(data))
        puck_pos_2d = get_init_puck_pos_smash_new()

        hit_angel = np.random.uniform(low=0, high=np.pi)
        hit_dir_2d = np.array([np.sin(hit_angel), np.cos(hit_angel)])
        # hit_dir_2d = goal_pos - puck_pos_2d
        # hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * mallet_radius

        q0, init_pos = get_init_configuration()
        if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
            print("Distance too short!")
            continue

        qd0 = get_initial_qd()
        qdd0 = np.zeros_like(q0)

        # qf2 = optimize2(np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]))
        # qdf2, v = get_qd_max_3d(np.array([*hit_dir_2d, 0]), qf2)
        # qf3 = optimize3(np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]))
        # qdf3, v3 = get_qd_max_3d(np.array([*hit_dir_2d, 0]), qf3)
        # qf4, v_opt = optimize4(np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]))
        # qdf4, v4 = get_qd_max_3d(np.array([*hit_dir_2d, v_opt]), qf4)

        # qf = optimize(np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]))
        # qdf, v1 = get_qd_max_3d(np.array([*hit_dir_2d, 0]), qf)
        # qddf = np.zeros_like(qf)
        # _qf = optimize(np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]))
        # _qdf, v1 = get_qd_max_3d(np.array([*hit_dir_2d, 0]), _qf)
        # qddf = np.zeros_like(_qf)

        # qf, qdf, success = optimize_vel_and_pos(np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]))
        qf, qdf, success = optimize_vel_and_pos_without_manipulability(np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]))
        qddf = np.zeros_like(qf)

        # x = np.array([v1, v, v3, v4])
        # sorted = np.argsort(x)
        # y = np.argmax(x)
        # if y in c:
        #     c[y] += 1
        #     c[y+4] += x[y] - x[sorted[-2]]
        # else:
        #     c[y] = 1
        #     c[y+4] = x[y] - x[sorted[-2]]

        # qf_limit = np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])
        # qdf_limit = 0.9 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
        # dq_bool = np.abs(qdf) > qdf_limit
        # q_bool = np.abs(qf) > qf_limit
        # print(dq_bool)
        # print(q_bool)
        # print(qdf[dq_bool])

        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, puck_pos_2d], axis=None))
        data.append(np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None))

    np.savetxt(f"hitting_data_{n}.tsv", data, delimiter='\t', fmt="%.10f")


## ===============================================================================
##                                  Validation
## ===============================================================================


def validate_if_initial_mallet_and_puck_positions_makes_hit_possible(xm, ym, xp, yp):
    """Validates if given initial mallet and puck positions enables one to plan reasonable movement"""
    dist = np.sqrt((ym - yp) ** 2 + (xm - xp) ** 2)
    return dist > 1.2 * (puck_radius + mallet_radius)


## ===============================================================================
##                                  Optimization
## ===============================================================================
def optimize_vel_and_pos(hitting_point, hitting_direction, scale=True):
    v = hitting_direction
    success_inv, initial_q = inverse_kinematics(robot_model, robot_data, hitting_point)
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

    constraints = [{'type': 'eq', 'fun': lambda q: FKp(q) - hitting_point}, {'type': 'ineq', 'fun': constraint_ineq}]
    result_q = minimize(hitting_objective, initial_q, args=(v,), constraints=constraints)
    if not result_q.success:
        optimal_q = result_q.x
    else:
        optimal_q = initial_q

    jacb_star = jacobian(robot_model, robot_data, optimal_q)[:3]
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
    result_eta = linprog(c=c, A_ub=A, b_ub=_ub, method='highs', x0=[0, 0, 0, 0, 0, 0, 0, 0], bounds=((0, None),
                         (dq_min[0],dq_max[0]), (dq_min[1],dq_max[1]), (dq_min[2],dq_max[2]), (dq_min[3],dq_max[3]),
                         (dq_min[4], dq_max[4]), (dq_min[5],dq_max[5]), (dq_min[6],dq_max[6])))
    if result_eta.success:
        optimal_eta = result_eta.x[0]
        optimal_alpha = result_eta.x[1:]
        optimal_v = optimal_eta * v
        if scale:
            epcilon = np.random.uniform()
        else:
            epcilon = 1
        _dq = (J_dag @ optimal_v + N @ optimal_alpha) * epcilon
        if (np.abs(_dq) > dq_max).any():
            _dq[np.abs(_dq) < 1e-5] = 1e-5
            beta = np.min(dq_max / np.abs(_dq))
            optimal_dq = _dq * beta
            print('beta_scale', )
        else:
            optimal_dq = _dq
    else:
        print('not success')
        if scale:
            epcilon = np.random.uniform()
        else:
            epcilon = 1
        qdf = np.linalg.lstsq(jacb_star, v, rcond=None)[0]
        max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
        optimal_dq = max_gain * qdf * epcilon
    return optimal_q, optimal_dq, result_eta.success

def optimize_vel_and_pos_without_manipulability(hitting_point, hitting_direction, scale=True):
    v = hitting_direction
    success_inv, initial_q = inverse_kinematics(robot_model, robot_data, hitting_point)
    dq_min = -0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
    dq_max = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])

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
    result_eta = linprog(c=c, A_ub=A, b_ub=_ub, method='highs', x0=[0, 0, 0, 0, 0, 0, 0, 0], bounds=((0, None),
                         (dq_min[0],dq_max[0]), (dq_min[1],dq_max[1]), (dq_min[2],dq_max[2]), (dq_min[3],dq_max[3]),
                         (dq_min[4], dq_max[4]), (dq_min[5],dq_max[5]), (dq_min[6],dq_max[6])))
    if result_eta.success:
        optimal_eta = result_eta.x[0]
        optimal_alpha = result_eta.x[1:]
        optimal_v = optimal_eta * v
        if scale:
            epcilon = np.random.uniform()
        else:
            epcilon = 1
        _dq = (J_dag @ optimal_v + N @ optimal_alpha) * epcilon
        if (np.abs(_dq) > dq_max).any():
            _dq[np.abs(_dq) < 1e-5] = 1e-5
            beta = np.min(dq_max / np.abs(_dq))
            optimal_dq = _dq * beta
            print('beta_scale', )
        else:
            optimal_dq = _dq
    else:
        print('not success')
        if scale:
            epcilon = np.random.uniform()
        else:
            epcilon = 1
        qdf = np.linalg.lstsq(jacb_star, v, rcond=None)[0]
        max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
        optimal_dq = max_gain * qdf * epcilon
    return initial_q, optimal_dq, result_eta.success


def optimize(hitting_point, hitting_direction):
    v = hitting_direction
    initial_q = inverse_kinematics(robot_model, robot_data, hitting_point)[1]

    def FKp(q):
        return forward_kinematics(robot_model, robot_data, q)[0]

    def hitting_objective(q, v):
        Jp = jacobian(robot_model, robot_data, q)[:3]
        manipulability = np.linalg.norm(np.dot(v, Jp))
        return -manipulability

    constraints = ({'type': 'eq', 'fun': lambda q: FKp(q) - hitting_point})
    result = minimize(hitting_objective, initial_q, args=(v,), constraints=constraints)
    optimal_q = result.x

    return optimal_q


def optimize2(hitting_point, hitting_direction):
    v = hitting_direction
    initial_q = inverse_kinematics(robot_model, robot_data, hitting_point)[1]

    def FKp(q):
        return forward_kinematics(robot_model, robot_data, q)[0]
    
    def hitting_constraint(q):
        return FKp(q) - hitting_point
    
    def z_constraint(q, v):
        Jp = jacobian(robot_model, robot_data, q)[:3]


    def hitting_objective(q, v):
        Jp = jacobian(robot_model, robot_data, q)[:2]
        manipulability = np.linalg.norm(np.dot(v[:2], Jp))
        return -manipulability

    constraints = ({'type': 'eq', 'fun': hitting_constraint})
    
    result = minimize(hitting_objective, initial_q, args=(v,), constraints=constraints)
    optimal_q = result.x

    return optimal_q


def optimize3(hitting_point, hitting_direction):
    initial_v = hitting_direction
    initial_q = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
    initial_params = np.array([*initial_q, initial_v[2]])

    def FKp(q):
        return forward_kinematics(robot_model, robot_data, q)[0]
    
    def hitting_constraint(q):
        return FKp(q) - hitting_point
    
    def z_constraint(q):
        return 0.001 - np.abs(initial_v[2])

    def hitting_objective(q):
        Jp = jacobian(robot_model, robot_data, q)[:2]
        manipulability = np.linalg.norm(np.dot(initial_v[:2], Jp))
        return -manipulability

    constraints = ({'type': 'eq', 'fun': hitting_constraint},
                   {'type': 'ineq', 'fun': z_constraint})
    
    result = minimize(hitting_objective, initial_q, constraints=constraints)
    optimal_q = result.x

    return optimal_q


def optimize4(hitting_point, hitting_direction):
    initial_v = hitting_direction
    initial_q = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
    initial_params = np.array([*initial_q, initial_v[2]])

    def FKp(q):
        return forward_kinematics(robot_model, robot_data, q)[0]
    
    def hitting_constraint(params):
        q, v = np.split(params, [len(initial_q)])
        return FKp(q) - hitting_point
    
    def z_constraint(params):
        q, v = np.split(params, [len(initial_q)])
        return 0.001 - np.abs(v[0])

    def hitting_objective(params):
        q, v = np.split(params, [len(initial_q)])

        v = np.array([*hitting_direction[:2], *v])
        Jp = jacobian(robot_model, robot_data, q)[:3]
        manipulability = np.linalg.norm(np.dot(v, Jp))
        return -manipulability

    constraints = ({'type': 'eq', 'fun': hitting_constraint},
                   {'type': 'ineq', 'fun': z_constraint})
    
    result = minimize(hitting_objective, initial_params, constraints=constraints)
    optimal_params = result.x

    return optimal_params[:len(initial_q)], optimal_params[-1]


def optimize_joint_angles_and_direction(hit_pos_3d, hit_dir_3d):
    # Define your forward kinematics function FKp(q) and hitting direction vector v
    def FKp(q):
        # Implement your forward kinematics function here
        # This function should return the position (x, y, z) for the given joint angles q
        return forward_kinematics(robot_model, robot_data, q)[0]

    def hitting_objective(params):
        q, v = np.split(params, [len(initial_q)])
        Jp = jacobian(robot_model, robot_data, q)[:3]  # Jacobian matrix at the hitting point
        manipulability = np.linalg.norm(np.dot(v, Jp))  # Manipulability along the hitting direction
        return -manipulability  # Negate to convert maximization problem to minimization

    # Example hitting direction vector
    initial_v = hit_dir_3d

    # Example initial guess for joint angles q
    initial_q = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])

    # Concatenate initial values
    initial_params = np.concatenate([initial_q, initial_v])

    # Define constraints for the optimization problem
    def direction_constraint(params):
        q, v = np.split(params, [len(initial_q)])
        return 0.001 - np.abs(v[2])  # Constraint: abs(optimal_direction[3]) < 0.01

    constraints = ({'type': 'eq', 'fun': lambda params: FKp(params[:len(initial_q)]) - hit_pos_3d},
                   {'type': 'ineq', 'fun': direction_constraint})

    # Perform optimization to find the optimal joint angles and hitting direction
    result = minimize(hitting_objective, initial_params, constraints=constraints)

    # Extract the optimal joint angles and hitting direction
    optimal_params = result.x
    optimal_q, optimal_direction = np.split(optimal_params, [len(initial_q)])

    # Print the optimal joint angles and hitting direction
    return optimal_q, optimal_direction


def load_train_data_replan():
    def load_data(filename='hitting_data_900.tsv'):
        path = os.path.join(os.path.abspath(os.getcwd()), "hitting_data_900.tsv")
        data = np.loadtxt(path, delimiter='\t').astype(np.float32)
        return torch.from_numpy(data)
    
    hitting_900 = load_data('hitting_data_900.tsv')
    hitting_100 = load_data('hitting_data_100.tsv')
    defend_450 = load_data('defend_data_450.tsv')
    defend_50 = load_data('defend_data_50.tsv')
    prepare_450 = load_data('prepare_data_450.tsv')
    prepare_50 = load_data('prepare_data_50.tsv')

    hitting = torch.from_numpy(np.concatenate([hitting_900, hitting_100]))
    defend =  torch.from_numpy(np.concatenate([defend_450, defend_50]))
    prepare = torch.from_numpy(np.concatenate([prepare_450, prepare_50]))

    return hitting, defend, prepare


def get_trajectory(q_cps, t_cps, device='cpu'):
    bsp_q = BSpline(num_pts=Config.bspline_q.n_ctr_pts, degree=Config.bspline_q.degree,
                        num_T_pts=Config.bspline_q.num_T_pts, device=device)
    bsp_t = BSpline(num_pts=Config.bspline_t.n_ctr_pts, degree=Config.bspline_t.degree,
                        num_T_pts=Config.bspline_t.num_T_pts, device=device)
    
    q = torch.einsum('ijk,lkm->ljm', bsp_q.N, q_cps)
    q_dot_tau = torch.einsum('ijk,lkm->ljm', bsp_q.dN, q_cps)
    q_ddot_tau = torch.einsum('ijk,lkm->ljm', bsp_q.ddN, q_cps)
    q_dddot_tau = torch.einsum('ijk,lkm->ljm', bsp_q.dddN, q_cps)

    dtau_dt = torch.einsum('ijk,lkm->ljm', bsp_t.N, t_cps)
    ddtau_dtt = torch.einsum('ijk,lkm->ljm', bsp_t.dN, t_cps)
    dddtau_dttt = torch.einsum('ijk,lkm->ljm', bsp_t.ddN, t_cps)

    dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]

    dtau_dt2 = dtau_dt ** 2
    q_dot = q_dot_tau * dtau_dt
    q_ddot = q_ddot_tau * dtau_dt2 + ddtau_dtt * q_dot_tau * dtau_dt
    q_dddot = q_dddot_tau * dtau_dt ** 3 + 3 * q_ddot_tau * ddtau_dtt * dtau_dt2 + \
            q_dot_tau * dtau_dt2 * dddtau_dttt + q_dot_tau * ddtau_dtt ** 2 * dtau_dt
        
    return q.cpu().detach().numpy(), q_dot.cpu().detach().numpy(), q_ddot.cpu().detach().numpy(), q_dddot.cpu().detach().numpy()


def create_replanning_dataset():
    hitting, defend, prepare = load_train_data_replan()

    device = 'cpu'
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model.pt")
    model = torch.load(model_path, map_location=torch.device(device))

    q_cps_hitting, t_cps_hitting = model(hitting[:, :42])
    q_cps_defend, t_cps_defend = model(defend[:, :42])
    q_cps_prepare, t_cps_prepare = model(prepare[:, :42])
    
    traj_hitting = get_trajectory(q_cps_hitting, t_cps_hitting)
    traj_defend = get_trajectory(q_cps_defend, t_cps_defend)
    traj_prepare = get_trajectory(q_cps_prepare, t_cps_prepare)

    def get_initial_sample(traj, dataset, margin=10):
        q, q_dot, q_ddot, _ = traj

        sample_id = np.random.randint(0, traj[0].shape[0])
        traj_point_id = np.random.randint(0, traj[0].shape[1] - 2 * margin) + margin
        sample_q = q[sample_id, traj_point_id]
        sample_q_dot = q_dot[sample_id, traj_point_id]
        sample_q_ddot = q_ddot[sample_id, traj_point_id]

        index = np.random.randint(0, dataset.shape[0])
        features_f = dataset[index, 21:]

        return np.concatenate([sample_q, sample_q_dot, sample_q_ddot, features_f])
    
    hitting, defend, prepare = hitting.cpu().detach().numpy(), defend.cpu().detach().numpy(), prepare.cpu().detach().numpy()
    data = []
    while len(data) < 9000:
        print(len(data))
        for dataset in [hitting, defend, prepare]:
            for traj in [traj_hitting, traj_defend , traj_prepare]:
                feature = get_initial_sample(traj, dataset)
                while not np.linalg.norm(forward_kinematics(robot_model, robot_data, feature[:7])[0][:2] - feature[-2:]) >= 0.085:
                    print("ERROR")
                    feature = get_initial_sample(traj, dataset)
                data.append(feature)

    n = len(data)
    np.savetxt(f"replanning_data_{int(.9  * n)}.tsv", data[:int(.9  * n)], delimiter='\t', fmt="%.10f")
    np.savetxt(f"replanning_data_{int(.1 * n)}.tsv", data[int(.9 * n):], delimiter='\t', fmt="%.10f")


def load_data_for_hitting_replan(file_name):
    def load_data(filename):
        path = os.path.join(os.path.abspath(os.getcwd()), filename)
        data = np.loadtxt(path, delimiter='\t').astype(np.float32)
        return torch.from_numpy(data)
    hitting = load_data(file_name)
    return hitting

def load_data_for_second_replan(file_name):
    def load_data(filename):
        path = os.path.join(os.path.abspath(os.getcwd()), filename)
        data = np.loadtxt(path, delimiter='\t').astype(np.float32)
        return torch.from_numpy(data)
    seconds = load_data(file_name)
    return seconds


def generate_first_replanning_dataset(n=100):
    hitting = load_data_for_hitting_replan('hitting_data_40000.tsv')

    device = 'cpu'
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/Model_7000.pt")
    model = torch.load(model_path, map_location=torch.device(device))

    q_cps_hitting, t_cps_hitting = model(hitting[:, :42])

    traj_hitting = get_trajectory(q_cps_hitting, t_cps_hitting)

    hitting = hitting.cpu().detach().numpy()
    data = []
    while len(data) < n:
        print(len(data))
        for dataset in [hitting]:
            for traj in [traj_hitting]:
                feature = get_replan_sample(traj, dataset)
                while not np.linalg.norm(
                        forward_kinematics(robot_model, robot_data, feature[:7])[0][:2] - feature[-2:]) >= 0.085:
                    print("ERROR")
                    feature = get_replan_sample(traj, dataset)
                data.append(feature)

    n = len(data)
    np.savetxt(f"first_replanning_data_{int(n)}.tsv", data[:int(n)], delimiter='\t', fmt="%.10f")


def generate_second_replanning_dataset(n=80):
    hitting = load_data_for_hitting_replan('hitting_data_40000.tsv')
    second_points = load_data_for_second_replan("first_replanning_data_100000.tsv")

    device = 'cpu'
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/Model_7000.pt")
    model = torch.load(model_path, map_location=torch.device(device))

    model_2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/Model_3015.pt")
    model_2 = torch.load(model_2_path, map_location=torch.device(device))

    q_cps_hitting, t_cps_hitting = model(hitting[:, :42])

    q_cps_seconds, t_cps_seconds = model_2(second_points[:, :42])


    traj_hitting = get_trajectory(q_cps_hitting, t_cps_hitting)

    traj_second = get_trajectory(q_cps_seconds, t_cps_seconds)

    hitting = hitting.cpu().detach().numpy()
    seconds = second_points.cpu().detach().numpy()
    data = []
    while len(data) < n:
        print(len(data))
        for dataset in [hitting, seconds]:
            for traj in [traj_hitting, traj_second]:
                feature = get_replan_sample(traj, dataset)
                while not np.linalg.norm(
                        forward_kinematics(robot_model, robot_data, feature[:7])[0][:2] - feature[-2:]) >= 0.085:
                    print("ERROR")
                    feature = get_replan_sample(traj, dataset)
                q_dot_in_limit, q_ddot_in_limit = validate_data(feature)
                while not (q_dot_in_limit and q_ddot_in_limit):
                    feature = get_replan_sample(traj, dataset)
                    q_dot_in_limit, q_ddot_in_limit = validate_data(feature)
                data.append(feature)

    n = len(data)
    np.savetxt(f"second_replanning_data_{int(n)}.tsv", data[:int(n)], delimiter='\t', fmt="%.10f")


def generate_third_replanning_dataset(n=80):
    hitting = load_data_for_hitting_replan('hitting_data_40000.tsv')
    second_points = load_data_for_second_replan("first_replanning_data_100000.tsv")
    third_points = load_data_for_second_replan("second_replanning_data_100000.tsv")

    device = 'cpu'
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/Model_7000.pt")
    model = torch.load(model_path, map_location=torch.device(device))

    model_2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/Model_3015.pt")
    model_2 = torch.load(model_2_path, map_location=torch.device(device))

    model_3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/Model_3015.pt")
    model_3 = torch.load(model_3_path, map_location=torch.device(device))

    q_cps_hitting, t_cps_hitting = model(hitting[:, :42])

    q_cps_seconds, t_cps_seconds = model_2(second_points[:, :42])

    q_cps_thirds, t_cps_thirds = model_3(third_points[:, :42])

    traj_hitting = get_trajectory(q_cps_hitting, t_cps_hitting)

    traj_second = get_trajectory(q_cps_seconds, t_cps_seconds)

    traj_third = get_trajectory(q_cps_thirds, t_cps_thirds)

    hitting = hitting.cpu().detach().numpy()
    seconds = second_points.cpu().detach().numpy()
    thirds = third_points.cpu().detach().numpy()
    data = []
    while len(data) < n:
        print(len(data))
        for dataset in [hitting, seconds, thirds]:
            for traj in [traj_hitting, traj_second, traj_third]:
                feature = get_replan_sample(traj, dataset)
                while not np.linalg.norm(
                        forward_kinematics(robot_model, robot_data, feature[:7])[0][:2] - feature[-2:]) >= 0.085:
                    print("ERROR")
                    feature = get_replan_sample(traj, dataset)
                q_dot_in_limit, q_ddot_in_limit = validate_data(feature)
                while not (q_dot_in_limit and q_ddot_in_limit):
                    feature = get_replan_sample(traj, dataset)
                    q_dot_in_limit, q_ddot_in_limit = validate_data(feature)
                data.append(feature)
    n = len(data)
    np.savetxt(f"third_replanning_data_{int(n)}.tsv", data[:int(n)], delimiter='\t', fmt="%.10f")


def get_replan_sample(traj, dataset):
    q, q_dot, q_ddot, _ = traj

    sample_id = np.random.randint(0, traj[0].shape[0])
    # hit or return?
    target = dataset[sample_id, -2:]
    start = 150
    end = 200
    if np.sum(target) < -100:
        start = 0
        end = 50
    traj_point_id = np.random.randint(start, end)

    sample_q = q[sample_id, traj_point_id]
    sample_q_dot = q_dot[sample_id, traj_point_id]
    sample_q_ddot = q_ddot[sample_id, traj_point_id]

    index = np.random.randint(0, dataset.shape[0])
    features_f = dataset[index, 21:]
    return np.concatenate([sample_q, sample_q_dot, sample_q_ddot, features_f])


q_dot_limits = Limits.q_dot7
q_ddot_limits = Limits.q_ddot7
huber = torch.nn.HuberLoss(reduction='none')


def validate_data(train_data):
    # q_dot = train_data[7:14]
    # q_ddot = train_data[14:21]
    # def compute_loss(loss_values, limits):
    #     loss_ = torch.relu(torch.abs(torch.tensor(loss_values)) - limits)
    #     loss_ = huber(loss_, torch.zeros_like(loss_))
    #     return loss_ * 0.1
    # q_dot_loss = torch.sum(compute_loss(q_dot, q_dot_limits))
    # q_ddot_loss = torch.sum(compute_loss(q_ddot, q_ddot_limits))
    # return q_dot_loss < 0.001, q_ddot_loss < 0.001
    return True, True


def generate_test_data():
    feature = np.array([0.3671,  -0.0504,  -0.3436,  -1.6178,  -0.2051,   1.2209,   0.0809,
          1.3606,   1.1785,  -1.3246,   1.2855,  -0.4356,   0.9063,  -0.1958,
          0.2530,   1.4681,   0.9956,  -0.8764,   2.0761,  -2.1426,  -0.5192,
        -13.4413,  -2.4849,  -4.4079, -20.8449,  29.2978,  -6.7165,   0.1211,
          1.3352,  -0.1441,  -0.2685,  -0.1747,   1.6011,   1.2298,   0.0000,
          0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
          0.8748,   0.0762])
    data = []
    while len(data) < 500:
        data.append(feature)
    np.savetxt('one_outlier.tsv', data, delimiter='\t', fmt="%.10f")


def generate_hit_pos_and_vel(n):
    data = []

    while len(data) < n:
        # print(len(data))
        puck_pos_2d = get_init_puck_pos_smash_new()

        hit_angel = np.random.uniform(low=0, high=np.pi)
        hit_dir_2d = np.array([np.sin(hit_angel), np.cos(hit_angel)])
        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * mallet_radius

        data.append(np.concatenate([hit_dir_2d, hit_pos_2d], axis=None))

    np.savetxt(f"pos_and_vel_data_{n}.tsv", data, delimiter='\t', fmt="%.10f")


def generate_uniform_data(n):
    data = []
    # complete hit
    for _ in range(int(n * 0.2)):
        puck_pos_2d = get_init_puck_pos_uniform()
        hit_angel = np.random.uniform(low=0, high=np.pi)
        hit_dir_2d = np.array([np.sin(hit_angel), np.cos(hit_angel)])
        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (mallet_radius + puck_radius)

        q0, init_pos = get_initial_config_plus_noise()
        if not validate_distance_make_hit_possible(init_pos[0], init_pos[1], puck_pos_2d[0], puck_pos_2d[1]):
            print("Distance too short!")
            continue

        qd0 = get_initial_joint_vel()
        qdd0 = np.zeros_like(q0)
        epcilon = np.random.uniform(0, 1)
        qf, qdf = compute_vel_and_pos(np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]), epcilon=epcilon)
        qddf = np.zeros_like(qf)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, puck_pos_2d], axis=None))
        # data.append(np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None))
    # start in hit range, uniform velocity, epcilon_0_1
    for _ in range(int(n * 0.6)):
        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_to_data(
                                                   start_point_function=get_init_puck_pos_uniform,
                                                   start_vel_low=-np.pi, start_vel_high=np.pi, start_epcilon_high=1,
                                                   start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
                                                   hit_epcilon_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))
    # start in hit range, towards backside

    # start in backside
    for _ in range(int(n * 0.05)):
        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_to_data(
                                                   start_point_function=get_middle_point_backside,
                                                   start_vel_low=0, start_vel_high=np.pi, start_epcilon_high=0.8,
                                                   start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
                                                   hit_epcilon_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))
        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_to_data(
                                                   start_point_function=get_middle_point_backside,
                                                   start_vel_low=-np.pi, start_vel_high=0, start_epcilon_high=0.5,
                                                   start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
                                                   hit_epcilon_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))
    # start in left and right side
    for _ in range(int(n * 0.05)):
        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_to_data(
                                                   start_point_function=get_middle_point_leftside,
                                                   start_vel_low=np.pi/2, start_vel_high=np.pi, start_epcilon_high=0.8,
                                                   start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
                                                   hit_epcilon_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))

        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_to_data(
                                                   start_point_function=get_middle_point_rightside,
                                                   start_vel_low=0, start_vel_high=np.pi/2, start_epcilon_high=0.8,
                                                   start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
                                                   hit_epcilon_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))
    np.savetxt(f"uniform_data_{n}.tsv", data, delimiter='\t', fmt="%.10f")


def generate_uniform_points(n):
    data = []
    for _ in range(int(n)):
        q0, qd0, qdd0, pos_2d = generate_configuration_of_point(
                                                   point_function=get_uniform_pos,vel_low=-np.pi,
                                                   vel_high=np.pi, epcilon_high=1, use_init_q=True)
        data.append(np.concatenate([q0, qd0, qdd0, pos_2d], axis=None))
    np.savetxt(f"uniform_data_point_{n}.tsv", data, delimiter='\t', fmt="%.10f")


def generate_configuration_of_point(point_function, vel_low, vel_high, epcilon_high, use_init_q):
    pos_2d = point_function()
    if use_init_q:
        init_state = compute_initial_state_add_noise()
    else:
        init_state = None
    vel_angel = np.random.uniform(low=vel_low, high=vel_high)
    vel_dir_2d = np.array([np.sin(vel_angel), np.cos(vel_angel)])
    epcilon = np.random.uniform(0, epcilon_high)
    q0, qd0 = vel_and_pos(np.array([*pos_2d, DESIRED_HEIGHT]), np.array([*vel_dir_2d, 0.]),
                                  epcilon=epcilon, initial_q=init_state)
    qdd0 = np.zeros_like(q0)
    return q0, qd0, qdd0, pos_2d


def vel_and_pos(point_pos, point_vel, epcilon, initial_q=None):
    v = point_vel
    success_inv, initial_q = inverse_kinematics(robot_model, robot_data, point_pos, initial_q=initial_q)
    dq_min = -0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
    dq_max = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])

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
    result_eta = linprog(c=c, A_ub=A, b_ub=_ub, method='highs', x0=[0, 0, 0, 0, 0, 0, 0, 0], bounds=((0, None),
                         (dq_min[0], dq_max[0]), (dq_min[1], dq_max[1]), (dq_min[2], dq_max[2]), (dq_min[3], dq_max[3]),
                         (dq_min[4], dq_max[4]), (dq_min[5], dq_max[5]), (dq_min[6], dq_max[6])))
    if result_eta.success:
        optimal_eta = result_eta.x[0]
        optimal_alpha = result_eta.x[1:]
        # clip axis_y velocity near table edge.
        optimal_eta_y = clip_eta_y(optimal_eta, point_pos[1], point_vel[1])
        optimal_eta_x = clip_eta_x(optimal_eta, point_pos[0], point_vel[0])
        optimal_v = np.array([optimal_eta_x, optimal_eta_y, optimal_eta]) * v
        _dq = (J_dag @ optimal_v + N @ optimal_alpha) * epcilon
        if (np.abs(_dq) > dq_max).any():
            _dq[np.abs(_dq) < 1e-5] = 1e-5
            beta = np.min(dq_max / np.abs(_dq))
            optimal_dq = _dq * beta
        else:
            optimal_dq = _dq
    else:
        qdf = np.linalg.lstsq(jacb_star, v, rcond=None)[0]
        max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
        optimal_dq = max_gain * qdf * epcilon
    return initial_q, optimal_dq


def clip_eta_y(eta_y, point_pos_y, point_vel_y):
    dis = mallet_radius * 3
    point_dis = 0.47085 - np.abs(point_pos_y)
    ratio = point_dis / dis
    if ratio < 1:
        if point_pos_y > 0 and point_vel_y > 0:
            return np.clip(a=eta_y, a_min=0, a_max=ratio * 0.2)
        if point_pos_y < 0 and point_vel_y < 0:
            return np.clip(a=eta_y, a_min=0, a_max=ratio * 0.2)
        return eta_y
    else:
        return eta_y


def clip_eta_x(eta_x, point_pos_x, point_vel_x):
    dis = mallet_radius * 3
    point_dis = 1.51 + point_pos_x - 0.60
    ratio = point_dis / dis
    if ratio < 1:
        if point_vel_x < 0:
            return np.clip(a=eta_x, a_min=0, a_max=ratio * 0.2)
        else:
            return eta_x
    else:
        return eta_x


def generate_configuration_to_data(start_point_function, start_vel_low, start_vel_high, start_epcilon_high,
                                   start_use_init_q, hit_vel_low, hit_vel_high, hit_epcilon_high, hit_use_init_q):
    hit_pos_2d = get_init_puck_pos_uniform()
    start_pos_2d = start_point_function()
    while not validate_distance_make_hit_possible(start_pos_2d[0], start_pos_2d[1], hit_pos_2d[0], hit_pos_2d[1]):
        print("Distance too short!")
        hit_pos_2d = get_init_puck_pos_uniform()
    if start_use_init_q:
        init_state = compute_initial_state_add_noise()
    else:
        init_state = None
    vel_angel = np.random.uniform(low=start_vel_low, high=start_vel_high)
    vel_dir_2d = np.array([np.sin(vel_angel), np.cos(vel_angel)])
    # add noise to middle_pos? initial_q and mj_data.qpos?
    epcilon = np.random.uniform(0, start_epcilon_high)
    q0, qd0 = compute_vel_and_pos(np.array([*start_pos_2d, DESIRED_HEIGHT]), np.array([*vel_dir_2d, 0.]),
                                  epcilon=epcilon, initial_q=init_state)
    qdd0 = np.zeros_like(q0)

    hit_angel = np.random.uniform(low=hit_vel_low, high=hit_vel_high)
    hit_dir_2d = np.array([np.sin(hit_angel), np.cos(hit_angel)])
    epcilon = np.random.uniform(0, hit_epcilon_high)
    if hit_use_init_q:
        init_state = compute_initial_state_add_noise()
    else:
        init_state = None
    qf, qdf = compute_vel_and_pos(np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]),
                                  epcilon=epcilon, initial_q=init_state)
    qddf = np.zeros_like(qf)
    return q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d


def get_middle_config_plus_noise():
    q_limits = torch.tensor([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])

    q_up_limit_deg = [120, 80, 120, 110, 120, 3]
    q_low_limit_deg = [-120, -30, -120, -30, -120, -3]
    q_up_limit_rad = [round(np.deg2rad(angle), 3) for angle in q_up_limit_deg]
    q_low_limit_rad = [round(np.deg2rad(angle), 3) for angle in q_low_limit_deg]

    mean = 0
    std_dev = np.deg2rad(10)

    _init_state = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.]) + np.random.normal(mean, std_dev, size=(7,))
    _init_state[6] = np.random.normal(mean, np.deg2rad(1), size=(1,))
    init_state = np.clip(a_min=q_low_limit_rad, a_max=q_up_limit_rad, a=_init_state)

    init_pos = np.array([0.65, 0., DESIRED_HEIGHT]) + np.random.uniform(low=-0.05, high=0.05, size=(3,))
    init_pos[2] = DESIRED_HEIGHT + 0.005 * (2 * np.random.rand() - 1.)

    success, init_state = inverse_kinematics(robot_model,
                                             robot_data,
                                             init_pos,
                                             R.from_euler('xyz', [0, 5 / 6 * np.pi, 0]).as_matrix(),
                                             initial_q=init_state)
    assert success is True
    return init_state, init_pos


def get_initial_config_plus_noise():
    init_state = compute_initial_state_add_noise()

    init_pos = np.array([0.65, 0., DESIRED_HEIGHT]) + np.random.uniform(low=-0.05, high=0.05, size=(3,))
    init_pos[2] = DESIRED_HEIGHT + 0.005 * (2 * np.random.rand() - 1.)

    success, init_state = inverse_kinematics(robot_model,
                                             robot_data,
                                             init_pos,
                                             R.from_euler('xyz', [0, 5 / 6 * np.pi, 0]).as_matrix(),
                                             initial_q=init_state)
    assert success is True
    return init_state, init_pos


def compute_initial_state_add_noise():
    # q_limits = torch.tensor([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])
    q_up_limit_deg = [120, 80, 120, 110, 120, 120, 3]
    q_low_limit_deg = [-120, -30, -120, -30, -120, -120, -3]
    q_up_limit_rad = [round(np.deg2rad(angle), 3) for angle in q_up_limit_deg]
    q_low_limit_rad = [round(np.deg2rad(angle), 3) for angle in q_low_limit_deg]

    mean = 0
    std_dev = np.deg2rad(10)

    _init_state = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.]) + np.random.normal(mean, std_dev, size=(7,))
    _init_state[6] = np.random.normal(mean, np.deg2rad(1), size=(1,))
    init_state = np.clip(a_min=q_low_limit_rad, a_max=q_up_limit_rad, a=_init_state)
    return init_state


def get_uniform_pos():
    range = np.array([[0.60, 1.3], [-0.47085, 0.47085]])
    point = np.random.rand(2) * (range[:, 1] - range[:, 0]) + range[:, 0]
    return point


def get_init_puck_pos_uniform():
    # y_lim = table_width / 2 - puck_radius - 2 * mallet_radius
    # hit_range = np.array([[0.8, 1.3], [-y_lim, y_lim]])
    hit_range = np.array([[0.8, 1.3], [-0.39105, 0.39105]])
    puck_pos = np.random.rand(2) * (hit_range[:, 1] - hit_range[:, 0]) + hit_range[:, 0]
    return puck_pos


def get_middle_point_backside():
    start_range = np.array([[0.65, 0.8], [-0.39105, 0.39105]])
    start_pos = np.random.rand(2) * (start_range[:, 1] - start_range[:, 0]) + start_range[:, 0]
    return start_pos


def get_middle_point_leftside():
    left_range = np.array([[0.65, 1.3], [-0.47085, -0.39105]])
    left_start_pos = np.random.rand(2) * (left_range[:, 1] - left_range[:, 0]) + left_range[:, 0]
    return left_start_pos


def get_middle_point_rightside():
    right_range = np.array([[0.65, 1.3], [0.39105, 0.47085]])
    right_start_pos = np.random.rand(2) * (right_range[:, 1] - right_range[:, 0]) + right_range[:, 0]
    return right_start_pos


def get_initial_joint_vel(var=0.25):
    q_dot = np.zeros((7,)) + np.random.uniform(low=-var, high=var, size=(7,))
    q_dot[6] = 0
    return q_dot


def validate_distance_make_hit_possible(xm, ym, xp, yp):
    """Validates if given initial mallet and puck positions enables one to plan reasonable movement"""
    dist = np.sqrt((ym - yp) ** 2 + (xm - xp) ** 2)
    return dist > 2 * (puck_radius + mallet_radius)


def compute_vel_and_pos(hitting_point, hitting_direction, epcilon, initial_q=None):
    v = hitting_direction
    success_inv, initial_q = inverse_kinematics(robot_model, robot_data, hitting_point, initial_q=initial_q)
    dq_min = -0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
    dq_max = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])

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
    result_eta = linprog(c=c, A_ub=A, b_ub=_ub, method='highs', x0=[0, 0, 0, 0, 0, 0, 0, 0], bounds=((0, None),
                         (dq_min[0],dq_max[0]), (dq_min[1],dq_max[1]), (dq_min[2],dq_max[2]), (dq_min[3],dq_max[3]),
                         (dq_min[4], dq_max[4]), (dq_min[5],dq_max[5]), (dq_min[6],dq_max[6])))
    if result_eta.success:
        optimal_eta = result_eta.x[0]
        optimal_alpha = result_eta.x[1:]
        optimal_v = optimal_eta * v
        _dq = (J_dag @ optimal_v + N @ optimal_alpha) * epcilon
        if (np.abs(_dq) > dq_max).any():
            _dq[np.abs(_dq) < 1e-5] = 1e-5
            beta = np.min(dq_max / np.abs(_dq))
            optimal_dq = _dq * beta
        else:
            optimal_dq = _dq
    else:
        qdf = np.linalg.lstsq(jacb_star, v, rcond=None)[0]
        max_gain = np.min(Limits.q_dot7.cpu().detach().numpy() / np.abs(qdf))
        optimal_dq = max_gain * qdf * epcilon
    return initial_q, optimal_dq


if __name__ == '__main__':
    # generate_replanning_dataset()
    # create_replanning_dataset()
    # generate_data_smash_new(n=int(100000 * .9))
    # generate_data_defend_new(n=int(500 * .9))
    # generate_data_prepare_new(n=int(500 * .9))
    # generate_data_smash_new(n=int(100000 * .1))
    # generate_data_defend_new(n=int(500 * .1))
    # generate_data_prepare_new(n=int(500 * .1))

    generate_uniform_points(n=200)