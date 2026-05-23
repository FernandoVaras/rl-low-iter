import sys
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fssa.wrapper import FaultCaptureWrapper
from fssa.module import FSSAModule

# ===== CONFIGURACIÓN =====
ROBOT     = "Walker2d"
ROBOT_DIR = "walker"
ENV_ID    = "Walker2d-v4"
ALGORITMO = "PPO"
USE_GSDE  = False
N_EPISODIOS = 5    # episodios a simular
# =========================

NOMBRE      = f"{ROBOT}_{ALGORITMO}{'_gSDE' if USE_GSDE else ''}"
MODELO_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
              "vanilla_models", ROBOT_DIR, "results", NOMBRE)

print("🧪 Test end-to-end de FSSAModule")
print("=" * 60)

# Cargar modelo entrenado
modelo = PPO.load(MODELO_PATH)

# Crear entorno con wrapper
env = gym.make(ENV_ID)
env = FaultCaptureWrapper(env)

# Crear módulo FSSA
fssa = FSSAModule(env)

print(f"  na (actuadores): {fssa.na}")
print(f"  g inicial (sin fallo): {fssa.get_current_g()}")

# Simular varios episodios
for ep in range(N_EPISODIOS):
    print(f"\n▶️  Episodio {ep+1}")
    obs, info = env.reset()

    # Decaer g al inicio del episodio
    g_actual = fssa.step_episode()
    print(f"  g al inicio: {np.round(g_actual, 4)}")

    while True:
        action, _ = modelo.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            if "fault_window" in info:
                print(f"  💥 Fallo en paso: {info['fault_step']}")
                g_nuevo = fssa.process_fault(info["fault_window"])
                print(f"  g calculado tras fallo: {np.round(g_nuevo, 4)}")
            else:
                print(f"  Episodio terminado sin fallo registrado")
            break

# Resultados finales
print(f"\n📊 HISTORIAL:")
print(f"  Episodios totales:     {fssa.episode_count}")
print(f"  Fallos registrados:    {len(fssa.history['fault_episodes'])}")
print(f"  g por episodio (últimos 3):")
for g in fssa.history["g_per_episode"][-3:]:
    print(f"    {np.round(g, 4)}")

env.close()
print("\n" + "=" * 60)
print("✅ Test completado")