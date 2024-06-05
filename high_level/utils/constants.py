import torch


class TableConstraint:
    XLB = 0.58415
    YLB = -0.47085
    XRT = 1.51
    YRT = 0.47085
    Z = 0.1645


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
