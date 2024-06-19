import torch
import numpy as np
from torch.utils.data import DataLoader


def load_data(path, batch_size, device='cpu', shuffle=True):
    def unpack_data_boundaries(x, n=7):
        row, col = x.shape

        q0 = np.column_stack([x[:, :n - 1], np.zeros((row, 1))])
        qk = np.column_stack([x[:, n:2 * n - 1], np.zeros((row, 1))])
        xyth = x[:, 2 * n: 2 * n + 3]
        q_dot_0 = np.column_stack([x[:, 2 * n + 3: 3 * n + 2], np.zeros((row, 1))])
        q_ddot_0 = np.column_stack([x[:, 3 * n + 3: 4 * n + 2], np.zeros((row, 1))])
        q_dot_k = np.column_stack([x[:, 4 * n + 3: 5 * n + 2], np.zeros((row, 1))])
        puck_pose = x[:, -2:]

        return q0, qk, xyth, q_dot_0, q_dot_k, q_ddot_0, puck_pose

    dataset_path = path

    train_data = np.loadtxt(dataset_path, delimiter='\t').astype(np.float32)
    q0, qk, xyth, q_dot_0, q_dot_k, q_ddot_0, puck_pose = unpack_data_boundaries(train_data)
    train_data = torch.from_numpy(np.hstack((q0, q_dot_0, q_ddot_0, qk, q_dot_k, np.zeros_like(q_ddot_0)))).to(torch.float32).to(device)
    train_ds = DataLoader(train_data, batch_size, shuffle=shuffle,
                          generator=torch.Generator(device=device))

    val_data = np.loadtxt(dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)
    q0, qk, xyth, q_dot_0, q_dot_k, q_ddot_0, puck_pose = unpack_data_boundaries(val_data)
    val_data = torch.from_numpy(np.hstack((q0, q_dot_0, q_ddot_0, qk, q_dot_k, np.zeros_like(q_ddot_0)))).to(torch.float32).to(device)
    val_ds = DataLoader(val_data, batch_size, shuffle=shuffle,
                        generator=torch.Generator(device=device))

    return train_ds, val_ds


def get_hitting_data(path, batch_size, device='cpu', shuffle=True, split=.9):
    dataset_path = path

    data = np.loadtxt(dataset_path, delimiter='\t').astype(np.float32)

    #np.random.shuffle(data)
    print(len(data))

    total_length = len(data)
    split_index = int(split * total_length)

    train_data = torch.from_numpy(data[:split_index]).to(device)
    val_data = torch.from_numpy(data[split_index:]).to(device)

    train_ds = DataLoader(train_data, batch_size, shuffle=shuffle,
                          generator=torch.Generator(device=device))
    val_ds = DataLoader(val_data, batch_size, shuffle=shuffle,
                        generator=torch.Generator(device=device))

    return train_ds, val_ds


def get_resample_data(path, batch_size, device='cpu', shuffle=True, split=.9):
    dataset_path = path

    data = np.loadtxt(dataset_path, delimiter='\t').astype(np.float32)

    #np.random.shuffle(data)
    print(len(data))

    total_length = len(data)
    split_index = int(split * total_length)

    train_data = torch.from_numpy(data[:split_index]).to(device)
    val_data = torch.from_numpy(data[split_index:]).to(device)

    train_ds = DataLoader(train_data, batch_size, shuffle=shuffle, generator=torch.Generator(device=device))
    val_ds = DataLoader(val_data, batch_size, shuffle=shuffle, generator=torch.Generator(device=device))

    return train_ds, val_ds


def update_dataloader(dataloader, new_data, batch_size, device, shuffle=True):
    updated_data = torch.cat((dataloader.dataset.data, new_data), dim=0)
    train_ds = DataLoader(updated_data, batch_size, shuffle=shuffle, generator=torch.Generator(device=device))

    print(updated_data.size())

    return train_ds
