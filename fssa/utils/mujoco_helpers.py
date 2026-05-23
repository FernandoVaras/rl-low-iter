import numpy as np


def get_actuated_indices(model):
    """
    Devuelve los índices nv→na, mapeando qué DOFs del modelo
    corresponden a articulaciones actuadas.

    Walker2d: nv=9, na=6 (los primeros 3 son la raíz, no actuada)
    Ant:      nv=14, na=8
    Humanoid: nv=23, na=17
    """
    # actuator_trnid[:, 0] da el joint_id que controla cada actuador
    joint_ids = model.actuator_trnid[:, 0]

    # Para cada joint, su DOF inicial
    actuated_indices = np.array([
        model.jnt_dofadr[jid] for jid in joint_ids
    ])

    return actuated_indices

def get_tau_max(model):
    """
    Devuelve el torque máximo absoluto por actuador (shape: na).
    Usa actuator_forcerange si está definido, sino calcula gear × ctrlrange.
    """
    na = model.nu
    tau_max = np.zeros(na)

    for i in range(na):
        if model.actuator_forcelimited[i]:
            tau_max[i] = np.max(np.abs(model.actuator_forcerange[i]))
        else:
            gear = model.actuator_gear[i, 0]
            ctrl_max = np.max(np.abs(model.actuator_ctrlrange[i]))
            tau_max[i] = gear * ctrl_max
            print(f"⚠️  Actuador {i} no tiene forcerange definido, "
                  f"usando gear × ctrlrange ({tau_max[i]:.3f})")

    return tau_max