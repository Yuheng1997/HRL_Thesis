import numpy as np
import torch
from mushroom_rl.utils.replay_memory import ReplayMemory


class TReplayMemory(ReplayMemory):
    def __init__(self, initial_size, max_size, state_shape, action_shape, device):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device

        super().__init__(initial_size, max_size)
        self._add_save_attr(_can_terminate='pickle!',
                            state_shape='primitive',
                            action_shape='primitive',
                            device='primitive'
                            )

    def add(self, dataset, n_steps_return=1, gamma=1.):
        i = 0
        while i < len(dataset):
            self._states[self._idx] = dataset[i][0]
            self._actions[self._idx] = dataset[i][1]
            self._rewards[self._idx] = dataset[i][2]

            self._next_states[self._idx] = dataset[i][3]
            self._absorbing[self._idx] = dataset[i][4]
            self._last[self._idx] = dataset[i][5]

            self._can_terminate[self._idx] = dataset[i][6]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0
            i += 1

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        initial_value = -1000.
        initial_boolean = True
        s = torch.full((n_samples, *self.state_shape), initial_value, device=self.device)
        a = torch.full((n_samples, *self.action_shape), initial_value, device=self.device)
        r = torch.full((n_samples,), initial_value, device=self.device)
        ss = torch.full((n_samples, *self.state_shape), initial_value, device=self.device)
        ab = torch.full((n_samples,), initial_value, device=self.device)
        last = torch.full((n_samples,), initial_value, device=self.device)
        can_terminate = torch.full((n_samples,), initial_boolean, device=self.device)

        j = 0
        for i in np.random.randint(self.size, size=n_samples):
            s[j] = self._states[i]
            a[j] = self._actions[i]
            r[j] = self._rewards[i]
            ss[j] = self._next_states[i]
            ab[j] = self._absorbing[i]
            last[j] = self._last[i]
            can_terminate[j] = self._can_terminate[i]
            j += 1

        return s, a, r, ss, ab, last, can_terminate

    def reset(self):
        self._idx = 0
        self._full = False
        initial_value = -1000.
        initial_boolean = True

        self._states = torch.full((self._max_size, *self.state_shape), initial_value, device=self.device)
        self._actions = torch.full((self._max_size, *self.action_shape), initial_value, device=self.device)
        self._rewards = torch.full((self._max_size,), initial_value, device=self.device)
        self._next_states = torch.full((self._max_size, *self.state_shape), initial_value, device=self.device)
        self._absorbing = torch.full((self._max_size,), initial_value, device=self.device)
        self._last = torch.full((self._max_size,), initial_value, device=self.device)
        self._can_terminate = torch.full((self._max_size,), initial_boolean, device=self.device)
