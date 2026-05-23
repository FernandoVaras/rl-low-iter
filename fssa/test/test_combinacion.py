import sys
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO, SAC

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fssa.wrapper import FaultCaptureWrapper
from fssa.pieces.culpa import compute_culpa
from fssa.pieces.margen import compute_margen
from fssa.pieces.combinacion import compute_combinacion
from fssa.utils.mujoco_helpers import get_actuated_indices, get_tau_max
from fssa.utils.adjacency import build_adjacency, build_laplacian_normalized

# ===== CONFIGURACIÓN - CAMBIA AQUÍ =====
ROBOT     = "Humanoid"
ROBOT_DIR = "humanoid"
ENV_ID    = "Humanoid-v4"
ALGORITMO = "PPO"
USE_GSDE  = False
# =======================================

NOMBRE      = f"{ROBOT}_{ALGORITMO}{'_gSDE' if USE_GSDE else ''}"
MODELO_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
              "vanilla_models", ROBOT_DIR, "results", NOMBRE)
CSV_PATH    = os.path.join(os.path.dirname(__file__), f"combinacion_{NOMBRE}.csv")

print("🧪 Test end-to-end de combinacion.py (con y sin difusión)")
print("=" * 60)

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

# Datos del modelo
mj_model = env.unwrapped.model
critical_body_id = env.critical_body_id
actuated_indices = get_actuated_indices(mj_model)
tau_max = get_tau_max(mj_model)
A = build_adjacency(mj_model)
laplacian = build_laplacian_normalized(A)

print(f"  nv (DOFs):        {mj_model.nv}")
print(f"  na (actuadores):  {mj_model.nu}")
print(f"  Matriz adyacencia:\n{A.astype(int)}")

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

            # Calcular culpa y margen una sola vez
            print(f"\n⚙️  Ejecutando compute_culpa y compute_margen...")
            c = compute_culpa(
                window=window,
                model=mj_model,
                dt_step=env.dt,
                critical_body_id=critical_body_id,
                actuated_indices=actuated_indices
            )
            m = compute_margen(window=window, tau_max=tau_max)

            # Calcular combinacion SIN difusión
            g_sin = compute_combinacion(c, m, use_diffusion=False)

            # Calcular combinacion CON difusión
            g_con = compute_combinacion(c, m,
                                        laplacian=laplacian,
                                        use_diffusion=True)

            # Resultados
            print(f"\n📊 RESULTADOS:")
            print(f"  Vector c:           {np.round(c, 4)}")
            print(f"  Vector m:           {np.round(m, 4)}")
            print(f"  Vector g (sin dif): {np.round(g_sin, 4)}")
            print(f"  Vector g (con dif): {np.round(g_con, 4)}")
            print(f"  Diferencia:         {np.round(g_con - g_sin, 4)}")

            # Validación
            print(f"\n✅ VALIDACIÓN:")
            print(f"  Ambos en [1.0, 2.0]: {np.all((g_sin >= 1) & (g_sin <= 2)) and np.all((g_con >= 1) & (g_con <= 2))}")

            # Guardar CSV
            df = pd.DataFrame({
                "junta":      [f"junta_{i}" for i in range(len(c))],
                "culpa":      c,
                "margen":     m,
                "g_sin_dif":  g_sin,
                "g_con_dif":  g_con,
                "diferencia": g_con - g_sin
            })
            df.to_csv(CSV_PATH, index=False)
            print(f"\n💾 Guardado en {CSV_PATH}")
        else:
            print("⚠️ Fallo detectado pero no alcanzó el warmup mínimo")
        break

env.close()
print("\n" + "=" * 60)
print("✅ Test completado")