import numpy as np
import torch as th
from gymnasium import spaces
from typing import Optional, Generator, NamedTuple
from typing import Union


class ReinforceBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_log_prob: th.Tensor
    returns: th.Tensor
    states: th.Tensor


class ReinforceBuffer:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    states: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_agents: int,
        device: Union[th.device, str] = "mps",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0]
        self.device = device
        self.state_shape = (n_agents, *observation_space.shape)
        self.n_envs = n_envs
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32
        )
        self.states = np.zeros(
            (self.buffer_size, self.n_envs, *self.state_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def compute_returns(self) -> None:
        """
        Post-processing step: compute the Monte-Carlo returns from rewards.
        (Because it's Monte Carlo without value function, we don't bootstrap).
        """
        last_return = np.zeros(self.n_envs, dtype=np.float32)

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                # We assume no bootstrap for REINFORCE (MC) or treat end of buffer as end of ep
                next_non_terminal = np.ones(self.n_envs, dtype=np.float32)
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]

            last_return = self.rewards[step] + self.gamma * last_return * next_non_terminal
            self.returns[step] = last_return

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        log_prob: th.Tensor,
        state: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param log_prob: log probability of the action
            following the current policy.
        :param state: Global state for centralized critic
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.obs_shape, tuple):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape states for multiple envs
        if isinstance(self.state_shape, tuple):
            state = np.stack(state)

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.states[self.pos] = np.array(state)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.log_probs[self.pos] = np.array(log_prob)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        elif self.pos > self.buffer_size:
            raise RuntimeError("Buffer overflow: Cannot add more data to full buffer")

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[ReinforceBufferSamples, None, None]:
        assert (
            self.full
        ), "Buffer must be full before sampling. Call collect_rollouts first."
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "states",
                "actions",
                "log_probs",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> ReinforceBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.log_probs[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.states[batch_inds],
        )
        return ReinforceBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)
