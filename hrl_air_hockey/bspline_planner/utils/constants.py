import torch
import numpy as np
import mujoco
import os


class TableConstraint:
    XMin = 0.6
    YMin = -0.47085
    XMax = 1.31
    YMax = 0.47085
    Z = 0.1645

class RobotEnvInfo:
    DESIRED_HEIGHT = TableConstraint.Z
    mallet_radius = 0.04815
    puck_radius = 0.03165
    table_width = 1.038
    table_length = 1.948
    goal_width = 0.25
    robot_offset = np.array([-1.51, 0, -0.1])

    PUCK_DEFAULT_POS = [-100, -100]

    robot_model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/iiwa_only.xml"))
    robot_model.body('iiwa_1/base').pos = np.zeros(3)
    robot_data = mujoco.MjData(robot_model)

class Limits:
    q7 = torch.tensor([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])
    q = q7[:6]
    q_dot7 = 0.8 * torch.tensor([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562], dtype=torch.float32)
    q_dot = q_dot7[:6]
    q_ddot7 = 10. * q_dot7
    q_ddot7 = torch.min(torch.stack([q_ddot7, 20. * torch.ones((7,), dtype=torch.float32)], dim=-1), dim=-1)[0]
    q_ddot = q_ddot7[:6]
    tau7 = 0.8 * torch.tensor([320, 320, 176, 176, 110, 40, 40], dtype=torch.float32)
    tau = tau7[:6]
    q_dddot7 = 5 * q_ddot7
    q_dddot = q_dddot7[:6]

    @staticmethod
    def to_device(device):
        for name, param in Limits.__dict__.items():
            if torch.is_tensor(param):
                setattr(Limits, name, param.to(device))


class UrdfModels:
    striker = "iiwa_striker.urdf"
    iiwa = "iiwa.urdf"
