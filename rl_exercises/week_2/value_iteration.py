# file: week_2/value_iteration.py

from __future__ import annotations

from typing import Any, Tuple

import gymnasium
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class ValueIteration(AbstractAgent):
    """Agent that computes an optimal policy via Value Iteration."""

    def __init__(
        self,
        env: MarsRover | gymnasium.Env,
        gamma: float = 0.9,
        seed: int = 333,
        **kwargs: dict,
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped  # type: ignore
        super().__init__(**kwargs)

        self.env = env
        self.gamma = gamma
        self.seed = seed

        # 从环境里提取 MDP 成分
        self.S = getattr(env, "states", np.arange(env.observation_space.n))
        self.A = getattr(env, "actions", np.arange(env.action_space.n))
        self.T = env.transition_matrix  # shape (nS, nA, nS)
        self.R_sa = env.get_reward_per_action()  # shape (nS, nA)

        self.n_states = self.R_sa.shape[0]
        self.n_actions = self.R_sa.shape[1]

        # placeholders
        self.V = np.zeros(self.n_states, dtype=float)
        self.pi = np.zeros(self.n_states, dtype=int)
        self.policy_fitted = False

    def update_agent(self, *args: tuple[Any], **kwargs: dict) -> None:
        """Run value iteration and store the resulting V and π."""
        if self.policy_fitted:
            return

        V_opt, pi_opt = value_iteration(
            T=self.T,
            R_sa=self.R_sa,
            gamma=self.gamma,
            seed=self.seed,
        )

        self.V[:] = V_opt
        self.pi[:] = pi_opt
        self.policy_fitted = True

    def predict_action(
        self,
        observation: int,
        info: dict | None = None,
        evaluate: bool = False,
    ) -> tuple[int, dict]:
        """Choose action = π(observation)."""
        if not self.policy_fitted:
            self.update_agent()
        return int(self.pi[observation]), {}



def value_iteration(
    *,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    seed: int | None = None,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Value Iteration on a finite MDP."""
    nS, nA = R_sa.shape
    V = np.zeros(nS, dtype=float)
    pi = np.zeros(nS, dtype=int)

    while True:
        delta = 0.0
        for s in range(nS):
            # 计算所有动作的 Q 值
            q_sa = R_sa[s] + gamma * (T[s] @ V)
            v_new = np.max(q_sa)
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < epsilon:
            break

    # 最终 greedy 策略
    for s in range(nS):
        q_sa = R_sa[s] + gamma * (T[s] @ V)
        # argmax 返回第一个最大值位置，保证确定性
        pi[s] = int(np.argmax(q_sa))

    return V, pi
