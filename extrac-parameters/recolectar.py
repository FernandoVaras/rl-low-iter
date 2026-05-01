import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar modelo (cambia por tu XML)
model = mujoco.MjModel.from_xml_path("/home/fernando/rl_env/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets/walker2d.xml")
data = mujoco.MjData(model)

# Tiempo de simulación
steps = 2000

# Listas para guardar datos
time = []
torso_height = []
velocity_x = []
energy_total = []
com_x = []
com_z = []

for i in range(steps):
    mujoco.mj_step(model, data)

    # Tiempo
    time.append(data.time)

    # Altura del torso (body 0 normalmente, ajusta si necesitas)
    torso_height.append(data.xpos[1][2])  # z

    # Velocidad en x
    velocity_x.append(data.qvel[0])

    # Energía total
    energy_total.append(np.sum(data.energy))

    # Centro de masa
    com = data.subtree_com[0]
    com_x.append(com[0])
    com_z.append(com[2])

# Crear DataFrame
df = pd.DataFrame({
    "time": time,
    "torso_height": torso_height,
    "velocity_x": velocity_x,
    "energy": energy_total,
    "com_x": com_x,
    "com_z": com_z
})

# Guardar CSV
df.to_csv("mujoco_data.csv", index=False)

print("Datos guardados ✔")
