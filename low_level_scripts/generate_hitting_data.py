import numpy as np
from scipy.spatial.transform import Rotation as R
from hrl_air_hockey.bspline_planner.utils.constants import RobotEnvInfo
from hrl_air_hockey.bspline_planner.utils.kinematics import inverse_kinematics
from hrl_air_hockey.bspline_planner.utils.compute_vel_and_pos import compute_vel_and_pos

table_width = RobotEnvInfo.table_width
table_length = RobotEnvInfo.table_length
puck_radius = RobotEnvInfo.puck_radius
mallet_radius = RobotEnvInfo.mallet_radius
robot_model = RobotEnvInfo.robot_model
robot_data = RobotEnvInfo.robot_data
DESIRED_HEIGHT = RobotEnvInfo.DESIRED_HEIGHT


def generate_start_end_points(n):
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
        scale = np.random.uniform(0, 1)
        qf, qdf = compute_vel_and_pos(robot_model, robot_data, np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]),
                                      scale=scale)
        qddf = np.zeros_like(qf)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, puck_pos_2d], axis=None))
        # data.append(np.concatenate([qf, qdf, qddf, q0, qd0, qdd0, PUCK_DEFAULT_POS], axis=None))
    # start in hit range, uniform velocity, epcilon_0_1
    for _ in range(int(n * 0.6)):
        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_start_end_points(
            start_point_function=get_init_puck_pos_uniform,
            start_vel_low=-np.pi, start_vel_high=np.pi, start_scale_high=1,
            start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
            hit_scale_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))
    # start in hit range, towards backside

    # start in backside
    for _ in range(int(n * 0.05)):
        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_start_end_points(
            start_point_function=get_middle_point_backside,
            start_vel_low=0, start_vel_high=np.pi, start_scale_high=0.8,
            start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
            hit_scale_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))
        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_start_end_points(
            start_point_function=get_middle_point_backside,
            start_vel_low=-np.pi, start_vel_high=0, start_scale_high=0.5,
            start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
            hit_scale_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))
    # start in left and right side
    for _ in range(int(n * 0.05)):
        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_start_end_points(
            start_point_function=get_middle_point_leftside,
            start_vel_low=np.pi / 2, start_vel_high=np.pi, start_scale_high=0.8,
            start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
            hit_scale_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))

        q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d = generate_configuration_start_end_points(
            start_point_function=get_middle_point_rightside,
            start_vel_low=0, start_vel_high=np.pi / 2, start_scale_high=0.8,
            start_use_init_q=True, hit_vel_low=0, hit_vel_high=np.pi,
            hit_scale_high=1, hit_use_init_q=False)
        data.append(np.concatenate([q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d], axis=None))
    np.savetxt(f"uniform_data_{n}.tsv", data, delimiter='\t', fmt="%.10f")


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


def generate_configuration_start_end_points(start_point_function, start_vel_low, start_vel_high, start_scale_high,
                                   start_use_init_q, hit_vel_low, hit_vel_high, hit_scale_high, hit_use_init_q):
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
    scale = np.random.uniform(0, start_scale_high)
    q0, qd0 = compute_vel_and_pos(robot_model, robot_data, np.array([*start_pos_2d, DESIRED_HEIGHT]), np.array([*vel_dir_2d, 0.]),
                                  scale=scale, initial_q=init_state)
    qdd0 = np.zeros_like(q0)

    hit_angel = np.random.uniform(low=hit_vel_low, high=hit_vel_high)
    hit_dir_2d = np.array([np.sin(hit_angel), np.cos(hit_angel)])
    scale = np.random.uniform(0, hit_scale_high)
    if hit_use_init_q:
        init_state = compute_initial_state_add_noise()
    else:
        init_state = None
    qf, qdf = compute_vel_and_pos(robot_model, robot_data, np.array([*hit_pos_2d, DESIRED_HEIGHT]), np.array([*hit_dir_2d, 0.]),
                                  scale=scale, initial_q=init_state)
    qddf = np.zeros_like(qf)
    return q0, qd0, qdd0, qf, qdf, qddf, hit_pos_2d



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

###########################################################################################
#
#          generate points method
#
###########################################################################################
def generate_uniform_points(n, save_path):
    data = []
    for _ in range(int(n)):
        q0, qd0, qdd0, pos_2d = generate_configuration_of_point(point_function=get_uniform_pos, vel_low=-np.pi,
                                                                vel_high=np.pi, scale_high=1, use_init_q=True)
        data.append(np.concatenate([q0, qd0, qdd0, pos_2d], axis=None))
    np.savetxt(f"{save_path}", data, delimiter='\t', fmt="%.10f")


def generate_configuration_of_point(point_function, vel_low, vel_high, scale_high, use_init_q):
    pos_2d = point_function()
    if use_init_q:
        init_state = compute_initial_state_add_noise()
    else:
        init_state = None
    vel_angel = np.random.uniform(low=vel_low, high=vel_high)
    vel_dir_2d = np.array([np.sin(vel_angel), np.cos(vel_angel)])
    scale = np.random.uniform(0, scale_high)
    q0, qd0 = compute_vel_and_pos(robot_model, robot_data, np.array([*pos_2d, DESIRED_HEIGHT]), np.array([*vel_dir_2d, 0.]),
                                  scale=scale, initial_q=init_state)
    qdd0 = np.zeros_like(q0)
    return q0, qd0, qdd0, pos_2d


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


if __name__ == '__main__':
    save_path = 'datasets/uniform_train/data.tsv'
    generate_uniform_points(n=10, save_path=save_path)
