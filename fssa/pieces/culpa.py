import numpy as np

# ===== HIPERPARÁMETROS - TODO: mover a config =====
LAMBDA_TEMPORAL = 0.1
W_QVEL = 0.4         # peso de anomalía de velocidad articular
W_DELTA_TAU = 0.6    # peso de anomalía de delta torque
EPSILON = 1e-8       # para evitar división por cero
# ==================================================


def compute_culpa(window, model, dt_step, critical_body_id, actuated_indices,
                  lambda_temporal=LAMBDA_TEMPORAL,
                  w_qvel=W_QVEL, w_delta_tau=W_DELTA_TAU):
    """
    Pieza 1+2+3: Calcula el vector de culpa por junta basado en anomalía multi-señal.

    Fórmula: c_i = w1 * |qvel_i^norm| + w2 * |delta_tau_i^norm|

    Donde:
        - qvel_i^norm:      velocidad articular normalizada por su máximo de ventana
        - delta_tau_i^norm: |ctrl_i - actuator_force_i| normalizado por su máximo de ventana

    Parámetros:
        window: dict con arrays shape (N, ...) del wrapper
        model: modelo MuJoCo (no usado en esta versión, mantenido por compatibilidad)
        dt_step: dt entre observaciones (no usado en esta versión)
        critical_body_id: id del body crítico (no usado en esta versión)
        actuated_indices: índices nv→na (para filtrar qvel)
        lambda_temporal: peso exponencial temporal
        w_qvel: peso para anomalía de velocidad
        w_delta_tau: peso para anomalía de delta torque

    Devuelve:
        c: np.array shape (na,) en [0,1], 1.0 = más culpable
    """
    N = window["qpos"].shape[0]
    na = len(actuated_indices)

    # Validación de dimensiones
    assert N == window["qvel"].shape[0], "qpos y qvel deben tener mismo N"
    assert na == model.nu, f"actuated_indices ({na}) ≠ model.nu ({model.nu})"
    assert np.isclose(w_qvel + w_delta_tau, 1.0), "Los pesos deben sumar 1"

    # ── Sub-paso 1: extraer y filtrar señales ──
    # qvel está en espacio nv, filtrar a na
    qvel_actuated = window["qvel"][:, actuated_indices]  # (N, na)
    qvel_abs = np.abs(qvel_actuated)                     # (N, na)

    # ctrl y actuator_force ya están en espacio na, NO filtrar
    delta_tau = window["ctrl"] - window["actuator_force"]  # (N, na)
    delta_tau_abs = np.abs(delta_tau)                       # (N, na)

    # ── Sub-paso 2: normalización por junta (cada junta su propia escala) ──
    # Máximo por junta en toda la ventana → shape (na,)
    qvel_max_per_joint = qvel_abs.max(axis=0)
    delta_tau_max_per_joint = delta_tau_abs.max(axis=0)

    # Normalizar (epsilon para evitar división por cero)
    qvel_norm = qvel_abs / (qvel_max_per_joint + EPSILON)             # (N, na)
    delta_tau_norm = delta_tau_abs / (delta_tau_max_per_joint + EPSILON)  # (N, na)

    # ── Sub-paso 3: combinar señales ponderadas ──
    c_t_array = w_qvel * qvel_norm + w_delta_tau * delta_tau_norm  # (N, na)

    # ── Sub-paso 4: agregación temporal con pesos exponenciales ──
    # t=N-1 es el último (más cercano al fallo) → peso máximo
    t_indices = np.arange(N)
    weights = np.exp(-lambda_temporal * (N - 1 - t_indices))
    weights = weights / weights.sum()

    c_aggregated = (weights[:, None] * c_t_array).sum(axis=0)  # (na,)

    # ── Sub-paso 5: normalización final por máximo ──
    c_max = c_aggregated.max()
    if c_max <= 0:
        return np.zeros(na)
    c_normalized = c_aggregated / c_max

    return c_normalized