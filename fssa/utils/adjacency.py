import numpy as np


def build_adjacency(model):
    """
    Construye la matriz de adyacencia entre actuadores basada en
    la cadena cinemática del modelo MuJoCo.

    Dos actuadores son vecinos si las juntas que controlan están
    conectadas en el árbol cinemático (comparten body padre o están
    a 1 body de distancia).

    Devuelve:
        A: matriz (na, na) con 1 si son vecinos, 0 si no
    """
    na = model.nu

    # Cada actuador controla una junta — esa junta está en un body
    actuator_bodies = np.zeros(na, dtype=int)
    for i in range(na):
        joint_id = model.actuator_trnid[i, 0]
        body_id = model.jnt_bodyid[joint_id]
        actuator_bodies[i] = body_id

    # Construir matriz de adyacencia
    A = np.zeros((na, na))

    for i in range(na):
        for j in range(na):
            if i == j:
                continue

            body_i = actuator_bodies[i]
            body_j = actuator_bodies[j]

            # Vecinos si comparten padre
            if model.body_parentid[body_i] == model.body_parentid[body_j]:
                A[i, j] = 1
                continue

            # Vecinos si uno es padre del otro
            if model.body_parentid[body_i] == body_j:
                A[i, j] = 1
                continue
            if model.body_parentid[body_j] == body_i:
                A[i, j] = 1
                continue

    return A


def build_laplacian_normalized(A):
    """
    Calcula el Laplaciano normalizado: L_norm = I - D^(-1/2) A D^(-1/2)
    Preserva la masa al aplicarse al vector g.
    """
    na = A.shape[0]
    degrees = A.sum(axis=1)

    # D^(-1/2), evitando división por cero
    d_inv_sqrt = np.zeros(na)
    nonzero = degrees > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(na) - D_inv_sqrt @ A @ D_inv_sqrt

    return L_norm