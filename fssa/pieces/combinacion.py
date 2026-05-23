import numpy as np
from .difusion import apply_diffusion

# ===== HIPERPARÁMETROS - TODO: mover a config =====
ALPHA = 0.7
KAPPA = 4.0
G_MIN = 1.0
G_MAX = 2.0
EPSILON = 1e-8
USE_DIFFUSION = False  # ← True para activar difusión cinemática
BETA = 0.2             # ← factor de propagación si está activa
# ==================================================


def compute_combinacion(c, m, alpha=ALPHA, kappa=KAPPA,
                        g_min=G_MIN, g_max=G_MAX,
                        laplacian=None, use_diffusion=USE_DIFFUSION, beta=BETA):
    """
    Pieza 5: Combina vector culpa y vector margen en factor de modulación g.

    Pipeline:
        1. g_raw = (c+ε)^α · (m+ε)^(1-α)
        2. g_pre = 1 + κ · (g_raw - mean(g_raw))
        3. (opcional) g_pre = (I + β·L_norm) · g_pre   ← difusión cinemática
        4. g_clipped = clip(g_pre, g_min, g_max)

    Parámetros:
        c: vector culpa  ∈ [0,1]^na
        m: vector margen ∈ [0,1]^na
        alpha, kappa, g_min, g_max: hiperparámetros estándar
        laplacian: matriz Laplaciano normalizado (si use_diffusion=True)
        use_diffusion: activa la difusión cinemática
        beta: factor de propagación para difusión

    Devuelve:
        g: np.array shape (na,) en [g_min, g_max]
    """
    assert c.shape == m.shape, f"c {c.shape} ≠ m {m.shape}"
    assert 0 <= alpha <= 1, f"alpha debe estar en [0,1], recibido {alpha}"

    # Sub-paso 1: combinación multiplicativa
    g_raw = (c + EPSILON) ** alpha * (m + EPSILON) ** (1 - alpha)

    # Sub-paso 2: re-escalado centrado en la media
    g_mean = g_raw.mean()
    g_pre = 1.0 + kappa * (g_raw - g_mean)

    # Sub-paso 3 (opcional): difusión cinemática
    if use_diffusion:
        assert laplacian is not None, "use_diffusion=True requiere laplacian"
        g_pre = apply_diffusion(g_pre, laplacian, beta)

    # Sub-paso 4: clip a [g_min, g_max]
    g_clipped = np.clip(g_pre, g_min, g_max)

    return g_clipped