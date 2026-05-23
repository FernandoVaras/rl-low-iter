import numpy as np

# ===== HIPERPARÁMETROS - TODO: mover a config =====
LAMBDA_TEMPORAL = 0.1
EPSILON = 1e-8
# ==================================================


def compute_margen(window, tau_max, lambda_temporal=LAMBDA_TEMPORAL):
    """
    Pieza 4: Calcula el vector de margen por motor.

    Fórmula:
        m_i^t = 1 - |τ_i^t| / τ_max_i
        m_aggregated = sum_t (w_t · m_i^t)
        m_clipped = clip(m_aggregated, 0, 1)

    Donde:
        - 1.0 = motor totalmente libre (sin uso)
        - 0.0 = motor saturado (al límite)

    Parámetros:
        window: dict con arrays del wrapper, requiere "actuator_force"
        tau_max: torque máximo absoluto por motor (na,) — de utils/mujoco_helpers.get_tau_max
        lambda_temporal: peso exponencial temporal

    Devuelve:
        m: np.array shape (na,) en [0,1]
    """
    N = window["actuator_force"].shape[0]
    na = window["actuator_force"].shape[1]

    # Validación
    assert tau_max.shape == (na,), f"tau_max shape ({tau_max.shape}) ≠ (na={na},)"

    # ── Sub-paso 1: margen instantáneo por timestep ──
    tau_abs = np.abs(window["actuator_force"])           # (N, na)
    saturation = tau_abs / (tau_max + EPSILON)            # (N, na)
    m_t = 1.0 - saturation                                # (N, na)

    # ── Sub-paso 2: agregación temporal exponencial ──
    t_indices = np.arange(N)
    weights = np.exp(-lambda_temporal * (N - 1 - t_indices))
    weights = weights / weights.sum()

    m_aggregated = (weights[:, None] * m_t).sum(axis=0)   # (na,)

    # ── Sub-paso 3: clip por seguridad numérica ──
    m_clipped = np.clip(m_aggregated, 0.0, 1.0)

    return m_clipped