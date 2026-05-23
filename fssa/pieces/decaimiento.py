import numpy as np

# ===== HIPERPARÁMETROS - TODO: mover a config =====
LAMBDA_EP = 0.3   # decaimiento medio: g cerca de 1 tras ~10 episodios
# ==================================================


def apply_decay(g_inicial, k, lambda_ep=LAMBDA_EP):
    """
    Pieza 7: Decaimiento exponencial episódico del vector g.

    Fórmula:
        g(k) = 1 + (g_inicial - 1) · exp(-λ_ep · k)

    Donde k = episodios desde el último fallo (k=0 → g = g_inicial).

    Parámetros:
        g_inicial: vector g calculado al momento del fallo, shape (na,)
        k: número de episodios desde el fallo (entero >= 0)
        lambda_ep: factor de decaimiento exponencial

    Devuelve:
        g_actual: vector decaído shape (na,)

    Ejemplo con lambda_ep=0.3 y g_inicial=2.0:
        k=0  → g = 2.0
        k=3  → g ≈ 1.41
        k=7  → g ≈ 1.12
        k=10 → g ≈ 1.05
    """
    assert k >= 0, f"k debe ser >= 0, recibido {k}"

    g_actual = 1.0 + (g_inicial - 1.0) * np.exp(-lambda_ep * k)
    return g_actual