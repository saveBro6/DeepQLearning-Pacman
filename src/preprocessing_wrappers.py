import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation
import ale_py
from config import CONFIG

def make_env(env_name, obs_type = "rgb", render_mode=None):
    # Setup Atari environment correctly with Farama Gymnasium wrappers
    gym.register_envs(ale_py)

    if obs_type == "ram":
        env = gym.make(env_name, obs_type="ram")
        env = FrameStackObservation(env, stack_size=CONFIG["FRAME_STACK"])
        return env 
    if render_mode is not None:
        env = gym.make(env_name, obs_type="rgb", render_mode=render_mode)
    else:
        env = gym.make(env_name, obs_type="rgb")
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=CONFIG["FRAME_STACK"])
    return env

