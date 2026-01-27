"""
CartPole Evaluation Module

Evaluates agents in the CartPole-v1 environment using gymnasium.

Usage:
    from benchmarks.cp.agents.cartpole_eval import compare_simple, angle_only
    agents = [('angle_only', angle_only())]
    compare_simple(agents)
"""

import time
import numpy as np

# Use gymnasium (successor to gym) - more actively maintained
try:
    import gymnasium as gym
    USE_GYMNASIUM = True
except ImportError:
    import gym
    USE_GYMNASIUM = False
    print("Warning: gymnasium not found, falling back to gym. Install with: pip install gymnasium")


def compare_simple(agents, n_episodes=100):
    """Compare agents over multiple episodes.

    Args:
        agents: List of (name, agent) tuples
        n_episodes: Number of episodes to run (default: 100)
    """
    for name, agent in agents:
        np.random.seed(0)
        env = gym.make('CartPole-v1')
        episode_rewards = [play_once(env, agent) for _ in range(n_episodes)]
        failcount = sum([1 for x in episode_rewards if x < 195])
        print(f'{name} \thad average episode rewards = {np.mean(episode_rewards):.1f}. Failed {failcount} times.')
        env.close()


def render_ntimes(agents, n, verbose=False, sleep=0):
    """Render agent behavior for visualization.

    Args:
        agents: List of (name, agent) tuples
        n: Number of episodes to render
        verbose: Print detailed output
        sleep: Delay between steps (seconds)
    """
    for name, agent in agents:
        np.random.seed(0)
        env.seed(0)
        env = gym.make('CartPole-v1', render_mode='human')
        for _ in range(n):
            episode_rewards = play_once(env, agent, render=True, verbose=verbose, sleep=sleep)
            print('Reward sum', episode_rewards)
        env.close()


def play_once(env, agent, render=False, verbose=False, sleep=0):
    """Play one episode with the given agent.

    Args:
        env: Gymnasium environment
        agent: Agent with decide(observation) method
        render: Whether to render (handled by env now)
        verbose: Print detailed output
        sleep: Delay between steps

    Returns:
        Total episode reward
    """
    # Handle both gymnasium and gym reset API
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        observation, info = reset_result  # gymnasium style
    else:
        observation = reset_result  # old gym style

    episode_reward = 0.
    if verbose:
        print('New agent')

    for step in range(2000):  # CartPole-v1 has max 500 steps
        action = agent.decide(observation, verbose=verbose)
        time.sleep(sleep)

        # Handle both gymnasium and gym step API
        step_result = env.step(action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            observation, reward, done, info = step_result

        episode_reward += reward

        if done:
            break

    return episode_reward


class angle_only:

    def decide(self, observation, verbose=False):
        cart_pos, cart_vel, pole_angle, pole_vel = observation[0], observation[1], observation[2], observation[3]
        if pole_angle < 0:
            return 0
        else:
            return 1


class poleVel_only:

    def decide(self, observation, verbose=False):
        cart_pos, cart_vel, pole_angle, pole_vel = observation[0], observation[1], observation[2], observation[3]
        if pole_vel < 0:
            return 0
        else:
            return 1


class SimonsFirst200:

    def decide(self, observation, verbose=False):
        cart_pos, cart_vel, pole_angle, pole_vel = observation[0], observation[1], observation[2], observation[3]

        if abs(pole_vel) > 0.1:
            if pole_vel < 0:
                return 0
            else:
                return 1

        # move_dir =
        if cart_vel < 0:
            return 0
        else:
            return 1


class SimonsBest:

    def decide(self, observation, verbose=False):
        cart_pos, cart_vel, pole_angle, pole_vel = observation[0], observation[1], observation[2], observation[3]

        if abs(pole_vel) > 0.1:
            if pole_vel < 0:
                return 0
            else:
                return 1

        if abs(cart_vel) > 0.1:
            if cart_vel < 0:
                return 0
            else:
                return 1

        if cart_pos > 0:
            return 0
        else:
            return 1


cartpole_agents = {1: ('angle_only', angle_only()),
                   2: ('poleVel_only', poleVel_only()),
                   3: ('SimonsFirst200', SimonsFirst200()),
                   4: ('SimonsBest', SimonsBest())}

oneAgent = {cartpole_agents[3]}
twoAgents = {cartpole_agents[4], cartpole_agents[3]}

# compare_simple(oneAgent)
render_ntimes(oneAgent, 1, verbose=True)
