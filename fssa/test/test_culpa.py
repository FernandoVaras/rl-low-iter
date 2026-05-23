import sys
import os
import numpy as np
import pandas as pd
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO, SAC

# Para importar fssa desde cualquier lugar
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fssa.wrapper import FaultCaptureWrapper
from fssa.pieces.culpa import compute_culpa
from fssa.utils.mujoco_helpers import get_actuated_indices

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
CSV_PATH    = os.path.join(os.path.dirname(__file__), f"culpa_{NOMBRE}.csv")

print("🧪 Test end-to-end de culpa.py (wrapper + culpa)")
print("=" * 50)

# Cargar modelo entrenado
print(f"📂 Cargando modelo: {MODELO_PATH}")
if ALGORITMO == "PPO":
    modelo = PPO.load(MODELO_PATH)
elif ALGORITMO == "SAC":
    modelo = SAC.load(MODELO_PATH)

# Crear entorno con wrapper
env = gym.make(ENV_ID)
env = FaultCaptureWrapper(env)

obs, info = env.reset()

# Datos del modelo para culpa
mj_model = env.unwrapped.model
critical_body_id = env.critical_body_id
actuated_indices = get_actuated_indices(mj_model)

print(f"  nv (DOFs):        {mj_model.nv}")
print(f"  na (actuadores):  {mj_model.nu}")
print(f"  actuated_indices: {actuated_indices}")
print(f"  critical_body_id: {critical_body_id}")

# Correr hasta el primer fallo
print(f"\n▶️  Corriendo episodio hasta el primer fallo...")
while True:
    action, _ = modelo.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        if "fault_window" in info:
            window = info["fault_window"]
            fault_step = info["fault_step"]
            print(f"  💥 Fallo detectado en paso: {fault_step}")
            print(f"  Pasos capturados en ventana: {window['qpos'].shape[0]}")

            # Calcular vector culpa
            print(f"\n⚙️  Ejecutando compute_culpa...")
            c = compute_culpa(
                window=window,
                model=mj_model,
                dt_step=env.dt,
                critical_body_id=critical_body_id,
                actuated_indices=actuated_indices
            )

            # Resultados
            print(f"\n📊 RESULTADOS:")
            print(f"  Vector c:  {np.round(c, 4)}")
            print(f"  Shape:     {c.shape}")
            print(f"  Junta más culpable: índice {int(np.argmax(c))} → valor {c.max():.4f}")

            # Validación
            print(f"\n✅ VALIDACIÓN:")
            print(f"  shape == (na,):  {c.shape == (mj_model.nu,)}")
            print(f"  todos en [0, 1]: {np.all((c >= 0) & (c <= 1))}")
            print(f"  c.max() == 1.0:  {np.isclose(c.max(), 1.0)}")

            # Guardar CSV
            df = pd.DataFrame({
                "junta": [f"junta_{i}" for i in range(len(c))],
                "culpa": c
            })
            df.to_csv(CSV_PATH, index=False)
            print(f"\n💾 Guardado en {CSV_PATH}")
        else:
            print("⚠️ Fallo detectado pero no alcanzó el warmup mínimo")
        break

env.close()
print("\n" + "=" * 50)
print("✅ Test completado")