import sys
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO, SAC

# Para importar el wrapper desde cualquier lugar
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from fssa.wrapper import FaultCaptureWrapper

# ===== CONFIGURACIÓN - CAMBIA AQUÍ =====
ROBOT     = "Walker2d"     # ← Walker2d, Ant, Humanoid
ROBOT_DIR = "walker"       # ← walker, ant, humanoid
ENV_ID    = "Walker2d-v4"  # ← Walker2d-v4, Ant-v4, Humanoid-v4
ALGORITMO = "PPO"          # ← PPO, SAC
USE_GSDE  = False          # ← True si usaste gSDE
# =======================================

NOMBRE      = f"{ROBOT}_{ALGORITMO}{'_gSDE' if USE_GSDE else ''}"
MODELO_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
              "vanilla_models", ROBOT_DIR, "results", NOMBRE)
CSV_PATH    = os.path.join(os.path.dirname(__file__), f"fault_window_{NOMBRE}.csv")

# Cargar modelo
print(f"📂 Cargando modelo: {MODELO_PATH}")
if ALGORITMO == "PPO":
    modelo = PPO.load(MODELO_PATH)
elif ALGORITMO == "SAC":
    modelo = SAC.load(MODELO_PATH)

# Crear entorno con wrapper
env = gym.make(ENV_ID)
env = FaultCaptureWrapper(env)

obs, info = env.reset()

# Correr hasta el primer fallo
while True:
    action, _ = modelo.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        if "fault_window" in info:
            window = info["fault_window"]
            N = window["qpos"].shape[0]

            # Aplanar todas las señales en un DataFrame
            rows = []
            for i in range(N):
                row = {"paso": i}
                row.update({f"qpos_{j}": window["qpos"][i][j]
                            for j in range(window["qpos"].shape[1])})
                row.update({f"qvel_{j}": window["qvel"][i][j]
                            for j in range(window["qvel"].shape[1])})
                row.update({f"actuator_force_{j}": window["actuator_force"][i][j]
                            for j in range(window["actuator_force"].shape[1])})
                row.update({f"ctrl_{j}": window["ctrl"][i][j]
                            for j in range(window["ctrl"].shape[1])})
                # cfrc_ext y subtree_com son 2D, aplanar
                row.update({f"cfrc_ext_{j}_{k}": window["cfrc_ext"][i][j][k]
                            for j in range(window["cfrc_ext"].shape[1])
                            for k in range(window["cfrc_ext"].shape[2])})
                row.update({f"subtree_com_{j}_{k}": window["subtree_com"][i][j][k]
                            for j in range(window["subtree_com"].shape[1])
                            for k in range(window["subtree_com"].shape[2])})
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(CSV_PATH, index=False)
            print(f"✅ Ventana pre-fallo guardada en {CSV_PATH}")
            print(f"   Pasos capturados: {N}")
            print(f"   Fallo en paso: {info['fault_step']}")
        else:
            print("⚠️ Fallo detectado pero no alcanzó el warmup mínimo")
        break

env.close()