from air_hockey_challenge.environments import position_control_wrapper as position
from mushroom_rl.utils.spaces import Box
import numpy as np
import time
import mujoco


class HitBackEnv(position.IiwaPositionTournament):
    def __init__(self, visual_target=False, horizon=9000, gamma=0.99, curriculum_steps=6, task_curriculum=True, initial_puck_pos=None):
        # 包括 定位goal，dynamics_info, info, mdp_info

        viewer_params = {
            'camera_params': {
                'static': dict(distance=3.0, elevation=-45.0, azimuth=90.0, lookat=(0., 0., 0.))
            }
        }
        super().__init__(gamma=gamma, horizon=horizon, interpolation_order=(-1, -1), viewer_params=viewer_params,
                         agent_name='agent',
                         opponent_name='opponent')
        self.visual_target = visual_target
        self.absorb_type = None
        self.info.gamma = gamma
        self.puck_pos = initial_puck_pos
        self.initial_puck_pos = None
        self.n_robot_joints = self.env_info['robot']["n_joints"]
        self.middle_timer = 0

        # start_side == -1, left agent serve
        self.start_side = -1
        self.prev_side = self.start_side

        # curriculum config
        self.start_range = None
        self.task_curriculum_dict = self.prepare_curriculum_dict(curriculum_steps)
        if task_curriculum:
            self.task_curriculum_dict['idx'] = 0
        else:
            self.task_curriculum_dict['idx'] = self.task_curriculum_dict['total_steps'] - 1
        # flag
        self.has_hit = False
        self.add_vel_bonus = False
        self.count_over_line = False
        self._absorbing = False
        self._task_success = False
        self.back_penalty = False
        self.hit_count = 0
        self.episode_end = False
        self.absorb_sign = False
        self.win = 0
        self.lose = 0
        self.not_cross_line = True

    def epoch_start(self):
        self.win = 0
        self.lose = 0
        self.hit_count = 0

    def prepare_curriculum_dict(self, curriculum_steps):
        curriculum_dict = {'total_steps': curriculum_steps}
        curriculum_dict['bonus_line'] = 0.5
        curriculum_dict['rate'] = np.linspace(0.9, 0.1, curriculum_dict['total_steps'])
        return curriculum_dict

    def is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)
        boundary = np.array([self.env_info['table']['length'], self.env_info['table']['width']]) / 2

        # Puck in Goal
        if (np.abs(puck_pos[1]) - self.env_info['table']['goal_width'] / 2) <= 0:
            if puck_pos[0] > self.env_info['table']['length'] / 2:
                self.score[0] += 1
                self.win += 1
                return True

            if puck_pos[0] < -self.env_info['table']['length'] / 2:
                self.score[1] += 1
                self.lose += 1
                self.absorb_sign = True
                return True

        if np.any(np.abs(puck_pos[:2]) > boundary) or np.linalg.norm(puck_vel) > 100:
            return True

        # Puck stuck in one side for more than 5s
        if np.sign(puck_pos[0]) == self.prev_side:
            self.side_timer += self.dt
        else:
            self.prev_side *= -1
            self.side_timer = 0

        if self.side_timer > 5.0 and np.abs(puck_pos[0]) >= 0.15:
            if self.prev_side == -1:
                self.absorb_sign = True
            return True

        # Puck stuck in the middle for 5s
        if np.abs(puck_pos[0]) < 0.15 and np.linalg.norm(puck_vel[0]) < 0.025 and self.middle_timer > 5.0:
            self.middle_timer = 0
            return True
        else:
            self.middle_timer += self.dt

        return self._absorbing

    def reward(self, obs, action, next_obs, absorbing):
        cur_puck_pos, cur_puck_vel = self.get_puck(obs)
        puck_pos, puck_vel = self.get_puck(next_obs)
        ee_pos, _ = self.get_ee()
        r = 0
        r_hit = 0
        r_cross = 0

        # check flag
        self.episode_end = False
        self._task_success = False
        if puck_pos[0] > 0.0 and puck_vel[0] < 0.0:
            if self.back_penalty:
                self.back_penalty = False
            if self.has_hit:
                if self.not_cross_line:
                    self.episode_end = True
                self.has_hit = False
            if self.add_vel_bonus:
                self.add_vel_bonus = False
            if not self.not_cross_line:
                self.not_cross_line = True

        # has_hit
        if not self.has_hit:
            if np.linalg.norm(puck_pos[:2] - ee_pos[:2]) - self.env_info['puck']['radius'] - self.env_info['mallet']['radius'] < 1e-2:
                r_hit += 10
                self.has_hit = True
                self.hit_count += 1

        if self.has_hit:
            if puck_vel[0] > 0.0:
                if not self.add_vel_bonus:
                    r_hit += puck_vel[0] * 30 + np.abs(puck_vel[1]) * 5
                    self.add_vel_bonus = True

        # penalty of backside
        if not self.back_penalty:
            if puck_pos[0] < -0.8:
                r -= 20
                self.back_penalty = True

        # reward of goal
        if (np.abs(puck_pos[1]) - self.env_info['table']['goal_width'] / 2) < 0:
            if puck_pos[0] > self.env_info['table']['length'] / 2:
                r += 200
            if puck_pos[0] < -self.env_info['table']['length'] / 2:
                r -= 200

        # success
        idx = self.task_curriculum_dict['idx']
        if self.not_cross_line:
            if puck_pos[0] > self.task_curriculum_dict['bonus_line'] and puck_vel[0] > 0.1:
                if self.has_hit:
                    self._task_success = True
                    self.not_cross_line = False
                    self.episode_end = True
                    r_cross = puck_vel[0] * 30
        if self.absorb_sign:
            self.episode_end = True
        rate = self.task_curriculum_dict['rate'][idx]
        r += r_hit * rate + r_cross * (1 - rate)
        return r

    def _create_info_dictionary(self, cur_obs):
        task_info = super()._create_info_dictionary(cur_obs)
        task_info['success'] = self._task_success
        task_info['win'] =  self.win
        task_info['lose'] = self.lose
        task_info['hit_num'] = self.hit_count
        task_info['sub_episodes'] = self.episode_end
        return task_info

    def step(self, action):
        if isinstance(action, list):
            low_action = action[0]
            return super().step(low_action)
        else:
            a1 = action[0].flatten()[:14].reshape(2, 7)
            a2 = action[1].flatten()[:14].reshape(2, 7)
            target_pos_2d = action[0].flatten()[14:16]
            if self.visual_target:
                self.update_visual_ball(target_pos_2d)
            return super().step((a1, a2))

    def setup(self, obs):
        self.absorb_sign = False
        self.not_cross_line = True
        self.episode_end = False
        self.add_vel_bonus = False
        self.back_penalty = False
        self.has_hit = False
        self._absorbing = False
        self._task_success = False
        self.side_timer = 0
        super().setup(obs)

        if self.start_side == -1:
            hit_range = np.array([[0.8 - 1.51, 1.3 - 1.51], [-0.39105, 0.39105]])
            puck_pos = np.random.rand(2) * (hit_range[:, 1] - hit_range[:, 0]) + hit_range[:, 0]
        else:
            opponent_range = np.array([[0.21, 0.71], [-0.39105, 0.39105]])
            puck_pos = np.random.rand(2) * (opponent_range[:, 1] - opponent_range[:, 0]) + opponent_range[:, 0]
        if self.puck_pos is not None:
            puck_pos = self.puck_pos

        self.start_side *= -1
        self.initial_puck_pos = puck_pos

        puck_yaw_pos = np.random.uniform(low=-np.pi, high=np.pi)
        puck_vel = np.zeros(3)
        puck_vel[2] = np.random.uniform(low=-2, high=2)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_yaw_pos", puck_yaw_pos)
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

    def update_task(self):
        if self.task_curriculum_dict['idx'] < self.task_curriculum_dict['total_steps'] - 1:
            self.task_curriculum_dict['idx'] += 1
            idx = self.task_curriculum_dict['idx']

    def update_task_vis(self, task_idx):
        self._model.site('puck_vis').type = mujoco.mjtGeom.mjGEOM_BOX
        self._model.site('puck_vis').rgba = np.array([0.3, 0.3, 0.3, 0.2])
        puck_range = self.task_curriculum_dict['puck_range'][task_idx]
        self._model.site('puck_vis').size = np.array([*(puck_range[1] - puck_range[0]) / 2, 0.001])
        self._model.site('puck_vis').pos = np.array([*(puck_range[1] + puck_range[0]) / 2, 0.0])

    def update_visual_ball(self, target_pos_2d):
        target_pos_2d = target_pos_2d - np.array([1.51, 0])
        self._model.site('ball_1').rgba = np.array([0.0, 0.6, 0.3, 0.2])
        self._model.site('ball_1').size = 0.05
        self._model.site('ball_1').pos = np.array([*target_pos_2d, 0.01])




if __name__ == '__main__':
    env = HitBackEnv(horizon=100, gamma=0.99)
    env.reset()

    count = 0
    steps = 0
    while True:
        action = np.zeros((2, 2, 7))
        steps += 1
        observation, reward, done, info = env.step(action)
        env.render()

        count += 1
        if count % 20 == 0:
            env.update_task()
            print(env.task_curriculum_dict['idx'])
            env.reset()
        time.sleep(0.01)
