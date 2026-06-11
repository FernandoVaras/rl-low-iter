import sys
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fssa.wrapper import FaultCaptureWrapper
from fssa.module import FSSAModule
from fssa.modulator import FSSAModulator

# ===== CONFIGURACIÓN =====
ROBOT     = "Walker2d"
ROBOT_DIR = "walker"
ENV_ID    = "Walker2d-v4"
ALGORITMO = "PPO"
USE_GSDE  = False
N_EPISODIOS = 3
# =========================

NOMBRE      = f"{ROBOT}_{ALGORITMO}{'_gSDE' if USE_GSDE else ''}"
MODELO_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
              "vanilla_models", ROBOT_DIR, "results", NOMBRE)

print("🧪 Test end-to-end de FSSAModulator")
print("=" * 60)

# Cargar modelo entrenado
modelo = PPO.load(MODELO_PATH)

# Pila de wrappers: env → FaultCaptureWrapper → FSSAModulator
env = gym.make(ENV_ID)
env = FaultCaptureWrapper(env)
fssa = FSSAModule(env)
env = FSSAModulator(env, fssa)

print(f"  na (actuadores): {fssa.na}")
print(f"  sigma_base:      {env.sigma_base}")

for ep in range(N_EPISODIOS):
    print(f"\n▶️  Episodio {ep+1}")
    obs, info = env.reset()
    g_actual = fssa.step_episode()
    print(f"  g al inicio: {np.round(g_actual, 4)}")

    pasos = 0
    while True:
        action, _ = modelo.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        pasos += 1

        if terminated:
            if "fault_window" in info:
                print(f"  💥 Fallo en paso: {info['fault_step']}")
                g_nuevo = fssa.process_fault(info["fault_window"])
                print(f"  g calculado tras fallo: {np.round(g_nuevo, 4)}")
            else:
                print(f"  Episodio terminado sin fallo")
            break
        if truncated:
            break

    # Estadística del ruido aplicado
    norms = [h["noise_norm"] for h in env.noise_history[-pasos:]]
    print(f"  Ruido promedio: {np.mean(norms):.4f}")
    print(f"  Pasos totales:  {pasos}")

print(f"\n📊 RESUMEN:")
print(f"  Episodios:        {N_EPISODIOS}")
print(f"  Steps registrados: {len(env.noise_history)}")
print(f"  Ruido promedio global: {np.mean([h['noise_norm'] for h in env.noise_history]):.4f}")

env.close()
print("\n" + "=" * 60)
print("✅ Test completado")