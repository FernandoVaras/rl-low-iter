import collections
import numpy as np
import gymnasium as gym
import mujoco

# ===== CONFIGURACIÓN - CAMBIA AQUÍ =====
window_config = {
    "N": 50,
    "K": 0,
    "warmup_min": 50,
    "critical_body": "torso"
}
# =======================================


class FaultCaptureWrapper(gym.Wrapper):
    """
    Wrapper genérico de captura pre-fallo.
    Monitorea cada paso y cuando detecta terminated=True + not is_healthy
    entrega los N pasos anteriores en info["fault_window"].
    """

    def __init__(self, env, config=None):
        super().__init__(env)

        cfg = config if config is not None else window_config
        self.N = cfg["N"]
        self.K = cfg["K"]
        self.warmup_min = cfg["warmup_min"]

        # dt real considerando frame-skip
        self.dt = (self.env.unwrapped.model.opt.timestep *
                   self.env.unwrapped.frame_skip)

        # Resolver body crítico
        self.critical_body_id = mujoco.mj_name2id(
            self.env.unwrapped.model,
            mujoco.mjtObj.mjOBJ_BODY,
            cfg.get("critical_body", "torso")
        )
        if self.critical_body_id == -1:
            raise ValueError(
                f"Cuerpo crítico '{cfg.get('critical_body', 'torso')}' "
                f"no encontrado en el modelo"
            )

        maxlen = self.N + self.K

        # Buffer circular en memoria — 8 señales
        self.buffer = {
            "qpos":           collections.deque(maxlen=maxlen),
            "qvel":           collections.deque(maxlen=maxlen),
            "cfrc_ext":       collections.deque(maxlen=maxlen),
            "subtree_com":    collections.deque(maxlen=maxlen),
            "actuator_force": collections.deque(maxlen=maxlen),
            "ctrl":           collections.deque(maxlen=maxlen),
            "subtree_linvel": collections.deque(maxlen=maxlen),
            "cvel":           collections.deque(maxlen=maxlen),
        }

        self.step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Acceder a los datos internos de MuJoCo
        data = self.env.unwrapped.data

        # Capturar las 8 señales primarias
        self.buffer["qpos"].append(data.qpos.copy())
        self.buffer["qvel"].append(data.qvel.copy())
        self.buffer["cfrc_ext"].append(data.cfrc_ext.copy())
        self.buffer["subtree_com"].append(data.subtree_com.copy())
        self.buffer["actuator_force"].append(data.actuator_force.copy())
        self.buffer["ctrl"].append(data.ctrl.copy())
        self.buffer["subtree_linvel"].append(data.subtree_linvel.copy())
        self.buffer["cvel"].append(data.cvel.copy())

        self.step_count += 1

        # Si hay fallo real y ya pasó el warmup
        is_healthy = getattr(self.env.unwrapped, 'is_healthy', True)
        if terminated and not is_healthy and self.step_count >= self.warmup_min:
            info["fault_window"] = self._extract_window()
            info["fault_step"] = self.step_count

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        for key in self.buffer:
            self.buffer[key].clear()
        return self.env.reset(**kwargs)

    def _extract_window(self):
        """
        Extrae la ventana pre-fallo del buffer.
        Si K=0 devuelve los últimos N pasos.
        Si K>0 devuelve los N pasos antes del offset K.
        """
        window = {}
        for key, deque in self.buffer.items():
            arr = np.array(deque)
            if self.K == 0:
                window[key] = arr[-self.N:]
            else:
                end = len(arr) - self.K
                start = max(0, end - self.N)
                window[key] = arr[start:end]
        return window