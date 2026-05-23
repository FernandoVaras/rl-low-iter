import numpy as np


def apply_diffusion(g, laplacian_normalized, beta=0.2):
    """
    Aplica difusión cinemática al vector g usando el Laplaciano normalizado.

    Fórmula: g_difundido = (I + β · L_norm) · g

    Esto propaga el valor de cada junta a sus vecinas en la cadena cinemática,
    capturando que juntas físicamente conectadas comparten responsabilidad.

    Parámetros:
        g: vector pre-cotas shape (na,)
        laplacian_normalized: L_norm de utils.adjacency.build_laplacian_normalized
        beta: factor de propagación (0.1-0.3 típico)

    Devuelve:
        g_difundido: shape (na,)
    """
    na = len(g)
    I = np.eye(na)
    g_difundido = (I + beta * laplacian_normalized) @ g
    return g_difundido