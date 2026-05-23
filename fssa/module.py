import numpy as np

from .pieces.culpa import compute_culpa
from .pieces.margen import compute_margen
from .pieces.combinacion import compute_combinacion
from .pieces.decaimiento import apply_decay
from .utils.mujoco_helpers import get_actuated_indices, get_tau_max
from .utils.adjacency import build_adjacency, build_laplacian_normalized


class FSSAModule:
    """
    FSSA Module — orquesta el cálculo y evolución del vector g.

    Pipeline:
        1. Wrapper detecta fallo y entrega ventana pre-fallo
        2. process_fault(window) calcula culpa → margen → combinación → g_inicial
        3. step_episode() aplica decaimiento exponencial cada nuevo episodio
        4. get_current_g() devuelve el g actual para el modulator
    """

    def __init__(self, env, config=None):
        """
        Parámetros:
            env: entorno con wrapper, expone unwrapped.model
            config: dict opcional con hiperparámetros que sobrescriben defaults
        """
        self.env = env
        self.model = env.unwrapped.model

        # Config con defaults
        cfg = config if config is not None else {}
        self.alpha = cfg.get("alpha", 0.7)
        self.kappa = cfg.get("kappa", 4.0)
        self.g_min = cfg.get("g_min", 1.0)
        self.g_max = cfg.get("g_max", 2.0)
        self.lambda_ep = cfg.get("lambda_ep", 0.3)
        self.use_diffusion = cfg.get("use_diffusion", False)
        self.beta = cfg.get("beta", 0.2)
        self.lambda_temporal = cfg.get("lambda_temporal", 0.1)
        self.w_qvel = cfg.get("w_qvel", 0.4)
        self.w_delta_tau = cfg.get("w_delta_tau", 0.6)

        # Extraer del modelo
        self.critical_body_id = env.critical_body_id
        self.actuated_indices = get_actuated_indices(self.model)
        self.tau_max = get_tau_max(self.model)
        self.na = self.model.nu

        # Laplaciano solo si se usa difusión
        self.laplacian = None
        if self.use_diffusion:
            A = build_adjacency(self.model)
            self.laplacian = build_laplacian_normalized(A)

        # Estado del módulo
        self.g_inicial = np.ones(self.na)   # antes del primer fallo: sin modulación
        self.g_actual = np.ones(self.na)
        self.k = 0                          # episodios desde el último fallo
        self.episode_count = 0              # contador global de episodios

        # Historial
        self.history = {
            "g_per_episode":       [],   # g aplicado cada episodio
            "g_inicial_per_fault": [],   # g calculado en cada fallo
            "fault_episodes":      [],   # números de episodio donde hubo fallo
        }

    def process_fault(self, window):
        """
        Procesa una ventana pre-fallo: calcula culpa, margen, combinación.
        Actualiza g_inicial y resetea k.
        """
        c = compute_culpa(
            window=window,
            model=self.model,
            dt_step=self.env.dt,
            critical_body_id=self.critical_body_id,
            actuated_indices=self.actuated_indices,
            lambda_temporal=self.lambda_temporal,
            w_qvel=self.w_qvel,
            w_delta_tau=self.w_delta_tau,
        )

        m = compute_margen(
            window=window,
            tau_max=self.tau_max,
            lambda_temporal=self.lambda_temporal,
        )

        g = compute_combinacion(
            c, m,
            alpha=self.alpha,
            kappa=self.kappa,
            g_min=self.g_min,
            g_max=self.g_max,
            laplacian=self.laplacian,
            use_diffusion=self.use_diffusion,
            beta=self.beta,
        )

        # Actualizar estado
        self.g_inicial = g.copy()
        self.g_actual = g.copy()
        self.k = 0

        # Historial
        self.history["g_inicial_per_fault"].append(g.copy())
        self.history["fault_episodes"].append(self.episode_count)

        return g

    def step_episode(self):
        """
        Llamar al inicio de cada nuevo episodio.
        Aplica decaimiento exponencial al g_actual y guarda en historial.
        """
        self.k += 1
        self.episode_count += 1

        self.g_actual = apply_decay(
            self.g_inicial, self.k, lambda_ep=self.lambda_ep
        )

        self.history["g_per_episode"].append(self.g_actual.copy())
        return self.g_actual

    def get_current_g(self):
        """Devuelve el g actual para que el modulator lo aplique."""
        return self.g_actual

    def reset(self):
        """Reinicia el módulo (útil para tests o cambios de seed)."""
        self.g_inicial = np.ones(self.na)
        self.g_actual = np.ones(self.na)
        self.k = 0
        self.episode_count = 0
        self.history = {
            "g_per_episode":       [],
            "g_inicial_per_fault": [],
            "fault_episodes":      [],
        }