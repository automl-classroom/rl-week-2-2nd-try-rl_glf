# file: week_2/policy_iteration.py

from __future__ import annotations

from typing import Any

import warnings

import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class PolicyIteration(AbstractAgent):
    """Agent that performs standard tabular policy iteration."""

    def __init__(
        self,
        env: MarsRover,
        gamma: float = 0.9,
        seed: int = 333,
        filename: str = "policy.npy",
        **kwargs: dict,
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped  # type: ignore
        super().__init__(**kwargs)

        self.env = env
        self.gamma = gamma
        self.seed = seed
        self.filename = filename

        # MDP 成分
        self.n_obs = env.observation_space.n
        self.n_actions = env.action_space.n
        self.S = np.arange(self.n_obs)
        self.A = np.arange(self.n_actions)
        self.T = env.transition_matrix
        self.R_sa = env.get_reward_per_action()

        # 初始 policy 和 Q
        self.pi = np.zeros(self.n_obs, dtype=int)
        self.Q = np.zeros((self.n_obs, self.n_actions), dtype=float)

        self.policy_fitted: bool = False
        self.steps: int = 0

    def predict_action(
        self, observation: int, info: dict | None = None, evaluate: bool = False
    ) -> tuple[int, dict]:
        """根据当前 policy 选动作。"""
        if not self.policy_fitted:
            self.update_agent()
        return int(self.pi[observation]), {}

    def update_agent(self, *args: tuple[Any], **kwargs: dict) -> None:
        """Run full policy iteration."""
        if self.policy_fitted:
            return

        Q_opt, pi_opt, n_steps = policy_iteration(
            Q=self.Q,
            pi=self.pi,
            MDP=(self.S, self.A, self.T, self.R_sa, self.gamma),
        )
        self.Q[:] = Q_opt
        self.pi[:] = pi_opt
        self.steps = n_steps
        self.policy_fitted = True

    def save(self, *args: tuple[Any], **kwargs: dict) -> None:
        if self.policy_fitted:
            np.save(self.filename, self.pi)
        else:
            warnings.warn("Tried to save policy but policy is not fitted yet.")

    def load(self, *args: tuple[Any], **kwargs: dict) -> np.ndarray:
        self.pi = np.load(self.filename)
        self.policy_fitted = True
        return self.pi


def policy_evaluation(
    pi: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """对固定 π 做迭代策略评估，返回 V."""
    nS = R_sa.shape[0]
    V = np.zeros(nS, dtype=float)

    while True:
        delta = 0.0
        for s in range(nS):
            a = pi[s]
            v_new = R_sa[s, a] + gamma * np.dot(T[s, a], V)
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < epsilon:
            break

    return V


def policy_improvement(
    V: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    nS, nA = R_sa.shape
    Q = np.zeros((nS, nA), dtype=float)
    pi_new = np.zeros(nS, dtype=int)

    for s in range(nS):
        for a in range(nA):
            Q[s, a] = R_sa[s, a] + gamma * np.dot(T[s, a], V)
        pi_new[s] = int(np.argmax(Q[s]))  

    return Q, pi_new


def policy_iteration(
    Q: np.ndarray,
    pi: np.ndarray,
    MDP: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, int]:
    S, A, T, R_sa, gamma = MDP
    steps = 0

    while True:
        # 1) 策略评估
        V = policy_evaluation(pi, T, R_sa, gamma, epsilon)
        # 2) 策略提升
        Q, pi_new = policy_improvement(V, T, R_sa, gamma)
        steps += 1
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new

    return Q, pi, steps


if __name__ == "__main__":
    algo = PolicyIteration(env=MarsRover())
    algo.update_agent()
    print("Converged in", algo.steps, "steps")
