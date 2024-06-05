import numpy as np

from mushroom_rl.core import Serializable


class PlusReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, initial_size, max_size):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain.

        """
        self._initial_size = initial_size
        self._max_size = max_size

        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _idx='primitive!',
            _full='primitive!',
            _states='pickle!',
            _actions='pickle!',
            _rewards='pickle!',
            _next_states='pickle!',
            _absorbing='pickle!',
            _last='pickle!',
            _smdp_length='pickle!'
        )

    def add(self, dataset, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """
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
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = [None for _ in range(self._max_size)]
        self._actions = [None for _ in range(self._max_size)]
        self._rewards = [None for _ in range(self._max_size)]
        self._next_states = [None for _ in range(self._max_size)]
        self._absorbing = [None for _ in range(self._max_size)]
        self._last = [None for _ in range(self._max_size)]
        self._smdp_length = [None for _ in range(self._max_size)]

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size > self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size

    def _post_load(self):
        if self._full is None:
            self.reset()