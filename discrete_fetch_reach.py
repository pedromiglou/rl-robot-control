import numpy as np
import gymnasium as gym

from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
from gymnasium.envs.registration import register

class MyEnv(MujocoFetchReachEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # convert actions to discrete
        values = [-0.2, 0, 0.2]
        self.discrete_actions = [(x,y,z,0) for x in values for y in values for z in values]
        self.action_space = gym.spaces.discrete.Discrete(len(self.discrete_actions))

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        
        action = np.array(self.discrete_actions[action], dtype=np.float32)

        # action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }

        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info


register(
    id=f"FetchReachDense-custom",
    entry_point="discrete_fetch_reach:MyEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)