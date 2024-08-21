from air_hockey_challenge.environments import position_control_wrapper as position
from mushroom_rl.utils.spaces import Box
import numpy as np
import mujoco


class BaseEnv(position.IiwaPositionTournament):
    def __init__(self, visual_target=False, horizon=9000, gamma=0.99, initial_puck_pos=None):
        viewer_params = {
            'camera_params': {
                'static': dict(distance=3.0, elevation=-45.0, azimuth=90.0, lookat=(0., 0., 0.))
            }
        }
        super().__init__(gamma=gamma, horizon=horizon, interpolation_order=(3, 3), viewer_params=viewer_params, agent_name='agent', opponent_name='opponent')
        self.visual_target = visual_target
        self.absorb_type = None
        self.gamma = gamma
        self.info.gamma = gamma
        self.puck_pos = initial_puck_pos
        self.initial_puck_pos = None
        self.left_edge_len = self.env_info['table']['length'] / 2 - self.env_info['puck']['radius']
        self.middle_edge_len = self.env_info['table']['width'] - 2 * self.env_info['puck']['radius']
        self.right_edge_len = self.env_info['table']['length'] / 2 - self.env_info['puck']['radius']
        self.total_len = self.left_edge_len + self.middle_edge_len + self.right_edge_len
        # self._goal_pos = np.random.uniform(low=0.0, high=self.total_len)
        self._goal_pos = 0
        self.n_robot_joints = self.env_info['robot']["n_joints"]
        self.cross_line_count = 0
        # flag
        self.has_hit = False
        self.count_over_line = False
        self._absorbing = False
        self._task_success = False
        self.back_penalty = False

    def is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)
        self._absorbing = super().is_absorbing(obs)
        if puck_pos[0] > 0.0:
            self._absorbing = True
        return self._absorbing

    def reward(self, obs, action, next_obs, absorbing):
        puck_pos, puck_vel = self.get_puck(next_obs)
        ee_pos, _ = self.get_ee()
        r = 0

        # has_hit
        if not self.has_hit:
            if puck_vel[0] > 0.1:
                self.has_hit = True
                v_norm = np.clip(puck_vel[0], a_min=0, a_max=2)
                r += v_norm * 30 + 30
                self._task_success = True
        # if self.has_hit:
        #     if puck_pos[0] > 0.1:
        #         v_norm = np.clip(puck_vel[0], a_min=0, a_max=2)
        #         r += v_norm * 20 + 10
        #         self._task_success = True
        return r

    def _create_info_dictionary(self, cur_obs):
        task_info = super()._create_info_dictionary(cur_obs)
        task_info['success'] = self._task_success
        # task_info['num_across_line'] = self.cross_line_count
        return task_info

    def step(self, action):
        if isinstance(action, list):
            low_action = action[0]
            return super().step(low_action)
        else:
            a1 = action[0].flatten()[:14].reshape(2, 7)
            a2 = action[1]
            angle = action[0].flatten()[14]
            target_pos = np.array([np.cos(angle), np.sin(angle)])/5 + np.array([1.51, 0])
            if self.visual_target:
                self.update_visual_ball(target_pos)
            return super().step((a1, a2))

    def setup(self, obs):
        self.back_penalty = False
        self.has_hit = False
        self.count_over_line = False
        self._absorbing = False
        self._task_success = False
        super().setup(obs)

        hit_range = np.array([[0.8-1.51, 1.3-1.51], [-0.39105, 0.39105]])
        if self.puck_pos is not None:
            puck_pos = self.puck_pos
        else:
            puck_pos = np.random.rand(2) * (hit_range[:, 1] - hit_range[:, 0]) + hit_range[:, 0]
        self.initial_puck_pos = puck_pos

        puck_vel = np.zeros(3)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])
        self.update_task_vis()

    def update_task_vis(self):
        self._model.site('puck_vis').type = mujoco.mjtGeom.mjGEOM_BOX
        self._model.site('puck_vis').rgba = np.array([0.3, 0.3, 0.3, 0.2])
        puck_range = np.array([[0.8-1.51, -0.39105], [1.3-1.51, 0.39105]])
        self._model.site('puck_vis').size = np.array([*(puck_range[1] - puck_range[0]) / 2, 0.001])
        self._model.site('puck_vis').pos = np.array([*(puck_range[1] + puck_range[0]) / 2, 0.0])

    def update_visual_ball(self, target_pos):
        self._model.site('ball_1').rgba = np.array([0.3, 0.9, 0.3, 0.2])
        self._model.site('ball_1').size = np.array(0.05)
        self._model.site('ball_1').pos = np.array([*target_pos, 0.0]) - np.array([1.51, 0, 0])

        self._model.site('ball_2').rgba = np.array([0.3, 0.9, 0.3, 0.2])
        self._model.site('ball_2').size = np.array(0.05)
        self._model.site('ball_2').pos = np.array([*target_pos, 0.0]) - np.array([1.51, 0, 0])


if __name__ == '__main__':
    env = BaseEnv()
    env.reset()

    steps = 0
    while True:
        action = np.zeros((2, 2, 7))
        steps += 1
        observation, reward, done, info = env.step(action)
        env.render()
        if done or steps > 20:
            steps = 0
            env.reset()