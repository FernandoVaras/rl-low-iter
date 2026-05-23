import sys
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO, SAC

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fssa.wrapper import FaultCaptureWrapper
from fssa.pieces.margen import compute_margen
from fssa.utils.mujoco_helpers import get_tau_max

# ===== CONFIGURACIÓN - CAMBIA AQUÍ =====
ROBOT     = "Walker2d"
ROBOT_DIR = "walker"
ENV_ID    = "Walker2d-v4"
ALGORITMO = "PPO"
USE_GSDE  = False
# =======================================

NOMBRE      = f"{ROBOT}_{ALGORITMO}{'_gSDE' if USE_GSDE else ''}"
MODELO_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
              "vanilla_models", ROBOT_DIR, "results", NOMBRE)
CSV_PATH    = os.path.join(os.path.dirname(__file__), f"margen_{NOMBRE}.csv")

print("🧪 Test end-to-end de margen.py (wrapper + margen)")
print("=" * 50)

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

# Calcular tau_max una vez
mj_model = env.unwrapped.model
tau_max = get_tau_max(mj_model)

print(f"  nv (DOFs):       {mj_model.nv}")
print(f"  na (actuadores): {mj_model.nu}")
print(f"  tau_max:         {np.round(tau_max, 3)}")

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

            # Calcular vector margen
            print(f"\n⚙️  Ejecutando compute_margen...")
            m = compute_margen(window=window, tau_max=tau_max)

            # Resultados
            print(f"\n📊 RESULTADOS:")
            print(f"  Vector m:  {np.round(m, 4)}")
            print(f"  Shape:     {m.shape}")
            print(f"  Motor más saturado: índice {int(np.argmin(m))} → margen {m.min():.4f}")
            print(f"  Motor más libre:    índice {int(np.argmax(m))} → margen {m.max():.4f}")

            # Validación
            print(f"\n✅ VALIDACIÓN:")
            print(f"  shape == (na,):  {m.shape == (mj_model.nu,)}")
            print(f"  todos en [0, 1]: {np.all((m >= 0) & (m <= 1))}")

            # Guardar CSV
            df = pd.DataFrame({
                "motor":      [f"motor_{i}" for i in range(len(m))],
                "margen":     m,
                "saturacion": 1.0 - m
            })
            df.to_csv(CSV_PATH, index=False)
            print(f"\n💾 Guardado en {CSV_PATH}")
        else:
            print("⚠️ Fallo detectado pero no alcanzó el warmup mínimo")
        break

env.close()
print("\n" + "=" * 50)
print("✅ Test completado")