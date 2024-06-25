import numpy as np
from mushroom_rl.utils.replay_memory import ReplayMemory


class SMDPReplayMemory(ReplayMemory):
    def __init__(self, initial_size, max_size):
        super().__init__(initial_size, max_size)
        self._add_save_attr(_smdp_length='pickle!')


    def add(self, dataset, n_steps_return=1, gamma=1.):
        assert n_steps_return > 0

        i = 0
        while i < len(dataset) - n_steps_return + 1:
            reward = dataset[i][2]
            j = 0
            while j < n_steps_return - 1:
                if dataset[i + j][5]:
                    i += j + 1
                    break
                j += 1
                reward += gamma ** j * dataset[i + j][2]
            else:
                self._states[self._idx] = dataset[i][0]
                self._actions[self._idx] = dataset[i][1]
                self._rewards[self._idx] = reward

                self._next_states[self._idx] = dataset[i + j][3]
                self._absorbing[self._idx] = dataset[i + j][4]
                self._last[self._idx] = dataset[i + j][5]

                self._smdp_length[self._idx] = dataset[i + j][6]

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
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        smdp_length = list()

        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            ab.append(self._absorbing[i])
            last.append(self._last[i])
            smdp_length.append(self._smdp_length[i])

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last), np.array(smdp_length)

    def reset(self):
        super().reset()
        self._smdp_length = [None for _ in range(self._max_size)]