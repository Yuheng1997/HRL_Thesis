import torch
import wandb
import numpy as np
from mushroom_rl.core import Logger

wandb_mode = 'online'
render = False
x_number_points = 40
y_number_points = 40
epcilons = np.array([1])
goal = np.array([2.49, 0])
dt = 0.02
agent_1 = 'Model_12000.pt'
config={'x_number_points':x_number_points,
        'y_number_points':y_number_points,
        'epcilons': epcilons,
        'goal' :goal,
        'dt' :dt,
        'agent_1':agent_1}


def relu(x):
    return np.maximum(0, x)

def obstacle_violation(q, hit_pos, forward_kin):
    ee_pos = forward_kin(q)[0]
    dist_from_puck = np.linalg.norm(hit_pos - ee_pos[:2])
    puck_loss = relu(0.0798 - dist_from_puck)
    return puck_loss


def boundary_violation(q, forward_kin):
    ee_pos = forward_kin(q)[0]
    x_b = [0.58415, 1.51]
    y_b = [-0.47085, 0.47085]
    z = 0.1645
    x_loss = relu(x_b[0] - ee_pos[0]) + relu(ee_pos[0] - x_b[1])
    y_loss = relu(y_b[0] - ee_pos[1]) + relu(ee_pos[1] - y_b[1])
    z_loss = relu(z - ee_pos[2]) + relu(ee_pos[2] - z)
    return x_loss, y_loss, z_loss


def dq_violation(dq):
    dq_limit = 0.9 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562])
    dq_loss = relu(np.abs(dq) - dq_limit)
    return dq_loss


def hit_time(traj):
    t = len(traj) * 0.02
    return t


def home_time(traj):
    t = len(traj) * 0.02
    return t


def hitted(obs):
    puck_pos_x = obs[0]
    if puck_pos_x > 1.61:
        return True
    else:
        return False


def sample_points_vel():
    points = []
    vels_dir = []
    hit_range = np.array([[0.8, 1.3], [-0.39105, 0.39105]])
    x_ndarray = np.linspace(start=0.8, stop=1.3, num=x_number_points, endpoint=True)
    y_ndarray = np.linspace(start=-0.39105, stop=0.39105, num=y_number_points, endpoint=True)
    for y in y_ndarray:
        for x in x_ndarray:
            points.append(np.array([x, y]))
            dir = goal - np.array([x, y])
            dir = dir / np.linalg.norm(dir)
            vels_dir.append(dir)
    return points, vels_dir


def sample_points_vel_for_render():
    points = []
    vels_dir = []
    hit_range = np.array([[0.8, 1.3], [-0.39105, 0.39105]])
    x_ndarray = np.linspace(start=0.8, stop=1.3, num=3, endpoint=True)
    y_ndarray = np.linspace(start=-0.39105, stop=0.39105, num=6, endpoint=True)
    for y in y_ndarray:
        for x in x_ndarray:
            points.append(np.array([x, y]))
            dir = goal - np.array([x, y])
            dir = dir / np.linalg.norm(dir)
            vels_dir.append(dir)
    return points, vels_dir


def sample_points_vel_for_heatmap(number_x, number_y):
    # number in [1, 2, 3, 4]
    points = []
    vels_dir = []
    start_x = 0.8 + (number_x - 1) * (1.3 - 0.8)/4
    end_x = 0.8 + number_x * (1.3 - 0.8)/4
    start_y = -0.39105 + (number_y - 1) * (0.39105 * 2) / 4
    end_y = -0.39105 + number_y * (0.39105 * 2) / 4
    x_ndarray = np.linspace(start=start_x, stop=end_x, num=10, endpoint=True)
    y_ndarray = np.linspace(start=start_y, stop=end_y, num=10, endpoint=True)
    for y in y_ndarray:
        for x in x_ndarray:
            points.append(np.array([x, y]))
            dir = goal - np.array([x, y])
            dir = dir / np.linalg.norm(dir)
            vels_dir.append(dir)
    return points, vels_dir



if __name__ == "__main__":
    import os
    from base_env import BaseEnv
    from double_low_agent import DoubleLowAgent

    env = BaseEnv()
    agent_2 = None
    agent = DoubleLowAgent(env.env_info, agent_1, agent_2)
    logger = Logger(log_name=f'{agent_1}_goal{goal}_e{epcilons}', results_dir='./test_logs', seed=0, use_timestamp=True, log_console=True)

    os.environ["WANDB_API_KEY"] = "a903361ff1d9498b25c276d46a0dcc63fe596aca"
    wandb.init(
        project='compare_nn_model',
        config=config,
        mode=wandb_mode
    )

    # points, vels_dir = sample_points_vel()
    # points, vels_dir = sample_points_vel_for_render()
    mallet_radius = 0.04815
    puck_radius = 0.03165

    # for i in range(len(points)):
    #     for epcilon in epcilons:
    #         obstacle_loss, x_loss, y_loss, z_loss, max_z_loss, dq_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    #         puck_position = points[i] + np.array([0.00001, 0.00001])
    #         v_dir = vels_dir[i] + np.array([0.00001, 0.00001])
    #         # puck_position = np.array([1.30001e+00, 1.00000e-05])
    #         # v_dir = np.array([1.00001e+00, 1.00000e-05])
    #         env.puck_pos = np.array(puck_position) - np.array([1.51, 0])
    #         hit_pos = puck_position - v_dir * (mallet_radius)
    #         obs = env.reset()
    #         agent.reset()
    #         agent.training_agent.epcilon = epcilon
    #         agent.training_agent.update_goal_for_analysis(goal=np.array([*hit_pos, *v_dir]))
    #
    #         hit_traj, hit_pos_traj, hit_vel_traj, home_traj, home_pos_traj, home_vel_traj = agent.training_agent.draw_traj_for_analysis()
    #         t_hit = hit_time(hit_traj)
    #         t_home = home_time(home_traj)
    #
    #         for act in hit_traj:
    #             u_1 = np.array(act).reshape(2, 7)
    #             u_2 = np.vstack([agent.training_agent.joint_anchor_pos, np.zeros_like(agent.training_agent.joint_anchor_pos)])
    #             u = np.array([u_1, u_2])
    #             obs, reward, done, info = env.step(u)
    #             if render:
    #                 env.render()
    #             obstacle_loss += obstacle_violation(obs[6:13], puck_position, agent.training_agent.forward_kinematics)
    #             _x_loss, _y_loss, _z_loss = boundary_violation(obs[6:13], agent.training_agent.forward_kinematics)
    #             x_loss += _x_loss * dt
    #             y_loss += _y_loss * dt
    #             z_loss += _z_loss * dt
    #             if _z_loss > max_z_loss:
    #                 max_z_loss = _z_loss
    #             dq_loss += dq_violation(obs[13:20]) * 0.02
    #         for act in home_traj:
    #             max_puck_vel_x = obs[0]
    #             u_1 = np.array(act).reshape(2, 7)
    #             u_2 = np.vstack([np.zeros_like(agent.training_agent.joint_anchor_pos), np.zeros_like(agent.training_agent.joint_anchor_pos)])
    #             u = np.array([u_1, u_2])
    #             obs, reward, done, info = env.step(u)
    #             if render:
    #                 env.render()
    #             _x_loss, _y_loss, _z_loss = boundary_violation(obs[6:13], agent.training_agent.forward_kinematics)
    #             x_loss += _x_loss * dt
    #             y_loss += _y_loss * dt
    #             z_loss += _z_loss * dt
    #             if _z_loss > max_z_loss:
    #                 max_z_loss = _z_loss
    #             dq_loss += dq_violation(obs[13:20]) * dt
    #         has_hit = hitted(obs)
    #         logger.epoch_info(i, epcilon=epcilon, puck_pos=np.round(puck_position,5), hitted=has_hit,
    #                           max_puck_vel_x=np.round(max_puck_vel_x, 5),
    #                           vel_direction=np.round(v_dir, 5), vel=np.round(v_dir * epcilon, 5), t_hit=np.round(t_hit, 5),
    #                           t_home=np.round(t_home, 5), x_loss=np.round(x_loss, 5),
    #                           y_loss=np.round(y_loss,5), z_loss=np.round(z_loss,5), dq0_loss=np.round(dq_loss[0], 5), dq1_loss=np.round(dq_loss[1], 5),
    #                           dq2_loss= np.round(dq_loss[2], 5), dq3_loss=np.round(dq_loss[3], 5),
    #                           dq4_loss=np.round(dq_loss[4], 5), dq5_loss=np.round(dq_loss[5], 5),
    #                           dq6_loss=np.round(dq_loss[6], 5),
    #                           obstacle_loss=np.round(obstacle_loss, 5))
    #
    #         wandb.log({'max_puck_vel_x': np.round(max_puck_vel_x, 5), 't_hit': np.round(t_hit, 5),
    #                    't_home': np.round(t_home, 5), 'x_loss': np.round(x_loss, 5),
    #                    'y_loss': np.round(y_loss,5), 'z_loss': np.round(z_loss,5),
    #                    'max_z_loss': np.round(max_z_loss, 5),
    #                    'dq0_loss': np.round(dq_loss[0], 5), 'dq1_loss': np.round(dq_loss[1], 5),
    #                    'dq2_loss': np.round(dq_loss[2], 5), 'dq3_loss': np.round(dq_loss[3], 5),
    #                    'dq4_loss': np.round(dq_loss[4], 5), 'dq5_loss': np.round(dq_loss[5], 5),
    #                    'dq6_loss': np.round(dq_loss[6], 5), 'obstacle_loss': np.round(obstacle_loss, 5)})
    # wandb.finish()

    logger_map = Logger(log_name=f'heatmap_{agent_1}_goal{goal}_e{epcilons}', results_dir='./test_logs', seed=0,
                    use_timestamp=True, log_console=True)
    logger_point = Logger(log_name=f'point_map_{agent_1}_goal{goal}_e{epcilons}', results_dir='./test_logs', seed=0,
                    use_timestamp=True, log_console=True)
    epoch = 1
    for i in range(1, 5):
        for j in range(1, 5):
            sum_length = 0
            obstacle_loss, x_loss, y_loss, z_loss, max_z_loss, dq_loss, max_x_loss, max_y_loss, sum_hit_t, sum_home_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            points, vels_dir = sample_points_vel_for_heatmap(i, j)
            for n in range(len(points)):
                for epcilon in epcilons:

                    puck_position = points[n] + np.array([0.00001, 0.00001])
                    v_dir = vels_dir[n] + np.array([0.00001, 0.00001])
                    env.puck_pos = np.array(puck_position) - np.array([1.51, 0])
                    hit_pos = puck_position - v_dir * (mallet_radius)
                    obs = env.reset()
                    agent.reset()
                    agent.training_agent.epcilon = epcilon
                    agent.training_agent.update_goal_for_analysis(goal=np.array([*hit_pos, *v_dir]))

                    hit_traj, hit_pos_traj, hit_vel_traj, home_traj, home_pos_traj, home_vel_traj = agent.training_agent.draw_traj_for_analysis()
                    t_hit = hit_time(hit_traj)
                    t_home = home_time(home_traj)
                    sum_hit_t += t_hit
                    sum_home_t += t_home

                    for act in hit_traj:
                        u_1 = np.array(act).reshape(2, 7)
                        u_2 = np.vstack([agent.training_agent.joint_anchor_pos,
                                         np.zeros_like(agent.training_agent.joint_anchor_pos)])
                        u = np.array([u_1, u_2])
                        obs, reward, done, info = env.step(u)
                        if render:
                            env.render()
                        obstacle_loss += obstacle_violation(obs[6:13], puck_position,
                                                            agent.training_agent.forward_kinematics) * dt
                        _x_loss, _y_loss, _z_loss = boundary_violation(obs[6:13],
                                                                       agent.training_agent.forward_kinematics)
                        x_loss += _x_loss * dt
                        y_loss += _y_loss * dt
                        z_loss += _z_loss * dt
                        if _z_loss > max_z_loss:
                            max_z_loss = _z_loss
                        if _x_loss > max_x_loss:
                            max_x_loss = _x_loss
                        if _y_loss > max_y_loss:
                            max_y_loss = _y_loss
                        dq_loss += dq_violation(obs[13:20]) * dt
                    for act in home_traj:
                        max_puck_vel_x = obs[0]
                        u_1 = np.array(act).reshape(2, 7)
                        u_2 = np.vstack([np.zeros_like(agent.training_agent.joint_anchor_pos),
                                         np.zeros_like(agent.training_agent.joint_anchor_pos)])
                        u = np.array([u_1, u_2])
                        obs, reward, done, info = env.step(u)
                        if render:
                            env.render()
                        _x_loss, _y_loss, _z_loss = boundary_violation(obs[6:13],
                                                                       agent.training_agent.forward_kinematics)
                        x_loss += _x_loss * dt
                        y_loss += _y_loss * dt
                        z_loss += _z_loss * dt
                        if _z_loss > max_z_loss:
                            max_z_loss = _z_loss
                        if _x_loss > max_x_loss:
                            max_x_loss = _x_loss
                        if _y_loss > max_y_loss:
                            max_y_loss = _y_loss
                        dq_loss += dq_violation(obs[13:20]) * dt
                    sum_length += len(hit_traj) + len(home_traj)
                    logger_point.epoch_info(epoch, i_j=[i, j], puck_pos=np.round(puck_position, 5),
                                            max_puck_vel_x=np.round(max_puck_vel_x, 5),
                                            vel_direction=np.round(v_dir, 5), vel=np.round(v_dir * epcilon, 5),
                                            t_hit=np.round(t_hit, 5),
                                            t_home=np.round(t_home, 5))
            # normalize
            x_loss = x_loss / len(points)
            y_loss = y_loss / len(points)
            z_loss = z_loss / len(points)
            dq_loss = dq_loss / len(points)
            mean_hit_t = sum_hit_t / len(points)
            mean_home_t = sum_home_t / len(points)
            obstacle_loss = obstacle_loss / len(points)
            logger_map.epoch_info(epoch, i_j=[i, j], max_x_loss=np.round(max_x_loss, 5), max_y_loss=np.round(max_y_loss, 5),
                                  max_z_loss=np.round(max_z_loss, 5),
                                  x_loss=np.round(x_loss, 5), y_loss=np.round(y_loss, 5), z_loss=np.round(z_loss, 5),
                                  obstacle_loss=np.round(obstacle_loss, 5),
                                  dq0_loss=np.round(dq_loss[0], 5),dq1_loss=np.round(dq_loss[1], 5),
                                  dq2_loss=np.round(dq_loss[2], 5), dq3_loss=np.round(dq_loss[3], 5),
                                  dq4_loss=np.round(dq_loss[4], 5), dq5_loss=np.round(dq_loss[5], 5),
                                  dq6_loss=np.round(dq_loss[6], 5),
                                  mean_hit_t=np.round(mean_hit_t, 5), mean_home_t=np.round(mean_home_t, 5),)
            wandb.log({ 'epoch':epoch, 'max_x_loss':np.round(max_x_loss, 5), 'max_y_loss':np.round(max_y_loss, 5),
                       'max_z_loss':np.round(max_z_loss, 5), 'x_loss': np.round(x_loss, 5),
                       'y_loss': np.round(y_loss, 5), 'z_loss': np.round(z_loss, 5),
                       'obstacle_loss': np.round(obstacle_loss, 5),
                       'dq0_loss': np.round(dq_loss[0], 5), 'dq1_loss': np.round(dq_loss[1], 5),
                       'dq2_loss': np.round(dq_loss[2], 5), 'dq3_loss': np.round(dq_loss[3], 5),
                       'dq4_loss': np.round(dq_loss[4], 5), 'dq5_loss': np.round(dq_loss[5], 5),
                       'dq6_loss': np.round(dq_loss[6], 5),
                       'mean_hit_t':np.round(mean_hit_t, 5), 'mean_home_t':np.round(mean_home_t, 5)})
            epoch += 1
    wandb.finish()