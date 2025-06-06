# Ignore "imported but unused"
# flake8: noqa: F401
import os
import warnings

import gymnasium as gym
import hydra
import numpy as np
import rl_exercises
from gymnasium.core import Env
from gymnasium.wrappers import TimeLimit
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from rich import print as printr

# 处理所有可能的导入错误
try:
    import compiler_gym
except ImportError:  # 明确指定异常类型
    warnings.warn("Could not import compiler_gym. Probably it is not installed.")
    compiler_gym = None

try:
    from minigrid.wrappers import FlatObsWrapper
except ImportError:
    warnings.warn("Could not import minigrid. Probably it is not installed.")
    # 创建假的FlatObsWrapper类
    class FlatObsWrapper:
        def __init__(self, *args, **kwargs):
            pass

# 处理stable_baselines3导入
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    warnings.warn("Could not import stable_baselines3. Probably it is not installed.")
    # 创建假的类
    class Monitor:
        def __init__(self, env, filename=None):
            self.env = env
        
        def reset(self, *args, **kwargs):
            return self.env.reset(*args, **kwargs)
        
        def step(self, *args, **kwargs):
            return self.env.step(*args, **kwargs)
        
        def close(self):
            return self.env.close()
    
    class PPO:
        pass
    
    class SAC:
        pass

from typing import Any, List, SupportsFloat
from functools import partial
from tqdm import tqdm

from rl_exercises.agent.abstract_agent import AbstractAgent
from rl_exercises.agent.buffer import SimpleBuffer
from rl_exercises.environments import MarsRover
from rl_exercises.week_2 import PolicyIteration, ValueIteration


def evaluate(env: gym.Env, agent: AbstractAgent, episodes: int = 100) -> float:
    """Evaluate a given Policy on an Environment.

    Parameters
    ----------
    env: gym.Env
        Environment to evaluate on
    agent: AbstractAgent
        Agent to evaluate
    episodes: int
        Evaluation episodes

    Returns
    -------
    float
        Mean evaluation rewards
    """
    episode_rewards: List[float] = []
    pbar = tqdm(total=episodes)
    for _ in range(episodes):
        obs, info = env.reset()
        episode_rewards.append(0)
        done = False
        episode_steps = 0
        while not done:
            action, _ = agent.predict_action(obs, info, evaluate=True)  # type: ignore[arg-type]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards[-1] += reward
            episode_steps += 1
            if terminated or truncated:
                done = True
                pbar.set_postfix(
                    {
                        "episode reward": episode_rewards[-1],
                        "episode step": episode_steps,
                    }
                )
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)


def make_env(env_name: str, env_kwargs: dict = {}) -> gym.Env:
    """Make environment based on name and kwargs.

    Parameters
    ----------
    env_name : str
        Environment name
    env_kwargs : dict, optional
        Optional env config, by default {}

    Returns
    -------
    gym.Env
        Instantiated env
    """
    if "compiler" in env_name:
        try:
            if compiler_gym is None:
                raise ImportError("compiler_gym not imported")
                
            from christmas_challenge.utils import ActionWrapper, SpaceWrapper
            
            benchmark = "cbench-v1/dijkstra"
            env = compiler_gym.make(
                "llvm-autophase-ic-v0",
                benchmark=benchmark,
                reward_space="IrInstructionCountNorm",
                # apply_api_compatibility=True,
            )

            # Pretend that we are using correct action and observation spaces to circumvent sb3's space checking
            # ⚠ This is horrible and hacky, please do not adopt this. This does not guarantuee that anything is working.
            env.action_space = SpaceWrapper(
                env.action_space, desired_space=gym.spaces.Discrete
            )
            env.observation_space = SpaceWrapper(
                env.observation_space, desired_space=gym.spaces.Box
            )
            env = ActionWrapper(env, int)

            env = TimeLimit(env, max_episode_steps=100)
        except ImportError:
            warnings.warn("Could not import compiler_gym related modules. Using MarsRover instead.")
            env = MarsRover(**env_kwargs)
    elif env_name == "MarsRover":
        env = MarsRover(**env_kwargs)
        # env = TimeLimit(env, max_episode_steps=env.horizon)
    elif "MiniGrid" in env_name:
        try:
            env = gym.make(env_name, **env_kwargs)
            # env = RGBImgObsWrapper(env)
            if FlatObsWrapper is not None:
                env = FlatObsWrapper(env)
        except Exception as e:
            warnings.warn(f"Error creating MiniGrid environment: {e}. Using MarsRover instead.")
            env = MarsRover(**env_kwargs)
    else:
        try:
            env = gym.make(env_name, **env_kwargs)
        except Exception as e:
            warnings.warn(f"Could not create environment {env_name}: {e}. Using MarsRover instead.")
            env = MarsRover(**env_kwargs)
    
    try:
        env = Monitor(env, filename="train")
    except Exception as e:
        warnings.warn(f"Could not wrap environment with Monitor: {e}")
    
    return env


@hydra.main("configs", "base", version_base="1.1")  # type: ignore[misc]
def train(cfg: DictConfig) -> float:
    """Train the agent.

    Parameters
    ----------
    cfg : DictConfig
        Agent/experiment configuration

    Returns
    -------
    float
        Mean return of n eval episodes

    Raises
    ------
    NotImplementedError
        _description_
    """
    env = make_env(cfg.env_name)
    printr(cfg)
    if cfg.agent == "sb3":
        return train_sb3(env, cfg)
    elif cfg.agent in ["policy_iteration", "value_iteration"]:
        agent = eval(cfg.agent_class)(env=env, **cfg.agent_kwargs)
    elif cfg.agent in ["tabular_q_learning", "vfa_q_learning", "dqn"]:
        policy_class = eval(cfg.policy_class)
        policy = partial(policy_class, **cfg.policy_kwargs)
        agent_class = eval(cfg.agent_class)
        agent = agent_class(env, policy, **cfg.agent_kwargs)
    else:
        # TODO: add your agent options here
        raise NotImplementedError

    buffer_cls = eval(cfg.buffer_cls)
    buffer = buffer_cls(**cfg.buffer_kwargs)
    state, info = env.reset()

    for step in range(cfg.training_steps):
        action, info = agent.predict_action(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        buffer.add(state, action, reward, next_state, (truncated or terminated), info)

        if len(buffer) > cfg.batch_size or (
            cfg.update_after_episode_end and (terminated or truncated)
        ):
            batch = buffer.sample(cfg.batch_size)
            agent.update_agent(batch)

        state = next_state

        if terminated or truncated:
            state, info = env.reset()

        if step % cfg.eval_every_n_steps == 0:
            eval_performance = evaluate(
                make_env(cfg.env_name, cfg.env_kwargs), agent, cfg.n_eval_episodes
            )
            print(f"Eval reward after {step} steps was {eval_performance}.")

    agent.save(str(os.path.abspath("model.csv")))
    final_eval = evaluate(env, agent, cfg.n_eval_episodes)
    print(f"Final eval reward was: {final_eval}")
    return final_eval


def train_sb3(env: gym.Env, cfg: DictConfig) -> float:
    """Train stablebaselines agent on env.

    Parameters
    ----------
    env : gym.Env
        Environment
    cfg : DictConfig
        Agent/experiment configuration

    Returns
    -------
    float
        Mean rewards
    """
    # 检查是否可以使用stable_baselines3
    if not ('PPO' in globals() and isinstance(PPO, type)):
        warnings.warn("Cannot run train_sb3 because stable_baselines3 is not available")
        return 0.0
        
    # Create agent
    model = eval(cfg.agent_class)(
        "MlpPolicy",
        env,
        verbose=cfg.verbose,
        tensorboard_log=cfg.log_dir,
        seed=cfg.seed,
        **cfg.agent_kwargs,
    )

    # Train agent
    model.learn(total_timesteps=cfg.total_timesteps)

    # Save agent
    model.save(cfg.model_fn)

    # Evaluate
    env = Monitor(gym.make(cfg.env_id), filename="eval")
    means = evaluate(env, model, episodes=cfg.n_eval_episodes)
    performance = np.mean(means)
    return performance


if __name__ == "__main__":
    train()
