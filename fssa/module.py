import numpy as np

from .pieces.culpa import compute_culpa
from .pieces.margen import compute_margen
from .pieces.combinacion import compute_combinacion
from .pieces.decaimiento import apply_decay
from .utils.mujoco_helpers import get_actuated_indices, get_tau_max
from .utils.adjacency import build_adjacency, build_laplacian_normalized


class FSSAModule:
    """
    FSSA Module — orquesta cálculo y evolución del vector g.

    Acepta config dict (formato YAML cargado) o None (usa defaults).
    """

    def __init__(self, env, config=None):
        self.env = env
        self.model = env.unwrapped.model

        # Cargar config flat o usar defaults
        cfg = config if config is not None else {}

        # Sección combinacion
        comb = cfg.get("combinacion", {})
        self.alpha = comb.get("alpha", 0.7)
        self.kappa = comb.get("kappa", 4.0)
        self.g_min = comb.get("g_min", 1.0)
        self.g_max = comb.get("g_max", 2.0)
        self.use_diffusion = comb.get("use_diffusion", False)
        self.beta = comb.get("beta", 0.2)

        # Sección decaimiento
        decay = cfg.get("decaimiento", {})
        self.lambda_ep = decay.get("lambda_ep", 0.3)

        # Sección culpa
        culpa = cfg.get("culpa", {})
        self.lambda_temporal = culpa.get("lambda_temporal", 0.1)
        self.w_qvel = culpa.get("w_qvel", 0.4)
        self.w_delta_tau = culpa.get("w_delta_tau", 0.6)

        # Extraer del modelo
        self.critical_body_id = env.critical_body_id
        self.actuated_indices = get_actuated_indices(self.model)
        self.tau_max = get_tau_max(self.model)
        self.na = self.model.nu

        # Laplaciano si se usa difusión
        self.laplacian = None
        if self.use_diffusion:
            A = build_adjacency(self.model)
            self.laplacian = build_laplacian_normalized(A)

        # Estado
        self.g_inicial = np.ones(self.na)
        self.g_actual = np.ones(self.na)
        self.k = 0
        self.episode_count = 0

        self.history = {
            "g_per_episode":       [],
            "g_inicial_per_fault": [],
            "fault_episodes":      [],
        }

    def process_fault(self, window):
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

        self.g_inicial = g.copy()
        self.g_actual = g.copy()
        self.k = 0

        self.history["g_inicial_per_fault"].append(g.copy())
        self.history["fault_episodes"].append(self.episode_count)

        return g

    def step_episode(self):
        self.k += 1
        self.episode_count += 1
        self.g_actual = apply_decay(
            self.g_inicial, self.k, lambda_ep=self.lambda_ep
        )
        self.history["g_per_episode"].append(self.g_actual.copy())
        return self.g_actual

    def get_current_g(self):
        return self.g_actual

    def reset(self):
        self.g_inicial = np.ones(self.na)
        self.g_actual = np.ones(self.na)
        self.k = 0
        self.episode_count = 0
        self.history = {
            "g_per_episode":       [],
            "g_inicial_per_fault": [],
            "fault_episodes":      [],
        }