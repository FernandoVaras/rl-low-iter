import numpy as np
import gymnasium as gym


# ===== DEFAULT - se usa si no se pasa config =====
SIGMA_BASE = 0.1
# =================================================


class FSSAModulator(gym.Wrapper):
    """
    Modulador de exploración FSSA.
    Acepta config dict con sección "modulator" o un sigma_base directo.
    """

    def __init__(self, env, fssa_module, config=None, sigma_base=None):
        super().__init__(env)
        self.fssa_module = fssa_module

        # Prioridad: argumento directo > config dict > default
        if sigma_base is not None:
            self.sigma_base = sigma_base
        elif config is not None and "modulator" in config:
            self.sigma_base = config["modulator"].get("sigma_base", SIGMA_BASE)
        else:
            self.sigma_base = SIGMA_BASE

        self.na = env.action_space.shape[0]
        self.noise_history = []

    def step(self, action):
        g = self.fssa_module.get_current_g()
        noise = np.random.normal(0, self.sigma_base, size=self.na) * g
        action_modulated = action + noise

        action_clipped = np.clip(
            action_modulated,
            self.env.action_space.low,
            self.env.action_space.high
        )

        self.noise_history.append({
            "g": g.copy(),
            "noise_norm": np.linalg.norm(noise),
        })

        return self.env.step(action_clipped)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)