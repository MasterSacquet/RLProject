# highway_env_wrapper.py

import gymnasium as gym
import highway_env  # obligatoire pour register l'env
from .shared_core_config import SHARED_CORE_CONFIG


class HighwayV0Env:
    def __init__(self, config_override=None, render_mode=None):
        """
        Wrapper pour highway-v0 avec config partagée
        """
        self.config = SHARED_CORE_CONFIG.copy()

        # Override config si besoin
        if config_override:
            self._deep_update(self.config, config_override)

        self.env = gym.make(
            "highway-v0",
            config=self.config,
            render_mode=render_mode
        )

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @staticmethod
    def _deep_update(base, updates):
        """
        Merge récursif des configs
        """
        for k, v in updates.items():
            if isinstance(v, dict) and k in base:
                HighwayV0Env._deep_update(base[k], v)
            else:
                base[k] = v