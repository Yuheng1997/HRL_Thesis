import torch

from utils.table import Table


table = Table()


def air_hockey_table(xyz, dt, huber):
    huber_along_path = lambda x: torch.sum(dt * huber(x, torch.zeros_like(x)), dim=-1)
    relu_huber_along_path = lambda x: huber_along_path(torch.relu(x))
    xlow_loss = relu_huber_along_path(table.xlb - xyz[..., 0])
    xhigh_loss = relu_huber_along_path(xyz[..., 0] - table.xrt)
    ylow_loss = relu_huber_along_path(table.ylb - xyz[..., 1])
    yhigh_loss = relu_huber_along_path(xyz[..., 1] - table.yrt)
    z_loss = huber_along_path(xyz[..., 2] - table.z)
    constraint_losses = torch.stack([xlow_loss, xhigh_loss, ylow_loss, yhigh_loss, z_loss], dim=-1)
    return constraint_losses, xlow_loss + xhigh_loss, ylow_loss + yhigh_loss, z_loss


def air_hockey_puck(xyz, dt, puck_pose):
    xy = xyz[..., :2]
    dist_from_puck = torch.sqrt(torch.sum((puck_pose[:, None] - xy) ** 2, axis=-1))
    puck_loss = torch.relu(0.0798 - dist_from_puck)
    idx_ = torch.argmin(puck_loss.flip(dims=[-1]), axis=-1)
    idx = (xyz.shape[1] - idx_.to(torch.float32)[:, None]) - 1
    range_ = torch.arange(xyz.shape[1], dtype=torch.float32)[None]
    threshold = torch.where(idx > range_, torch.ones_like(puck_loss), torch.zeros_like(puck_loss))

    puck_loss = torch.sum(puck_loss * threshold * dt, axis=-1)
    return puck_loss
