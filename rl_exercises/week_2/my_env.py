from __future__ import annotations

import gymnasium as gym
import numpy as np

# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.state = 0  # initial state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise RuntimeError(f"Invalid action {action}")
        
        self.state = action
        reward = float(action)
        terminated = False
        truncated = False
        return int(self.state), reward, terminated, truncated, {}
    
    def get_reward_per_action(self):
        return np.array([[0, 1],
                         [0, 1]])

    def get_transition_matrix(self):
        T = np.zeros((2, 2, 2))
        T[0, 0, 0] = 1.0
        T[0, 1, 1] = 1.0
        T[1, 0, 0] = 1.0
        T[1, 1, 1] = 1.0
        return T



class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        super().__init__(env)
        self.noise = noise
        self.np_random = np.random.default_rng(seed)
    
    def reset(self, **kwargs):
        true_obs, info = self.env.reset(**kwargs)
        return self._noisy_obs(true_obs), info
    
    def step(self, action):
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(true_obs), reward, terminated, truncated, info
    
    def _noisy_obs(self, true_obs: int) -> int:
        if self.rng.random() < self.noise:
            n = self.observation_space.n
            others = [s for s in range(n) if s != true_obs]
            return int(self.rng.choice(others))
        else:
            return int(true_obs)
