import sys
import os
import gymnasium as gym
from stable_baselines3 import PPO

# ===== CONFIGURACIÓN - CAMBIA AQUÍ =====
USE_GSDE = False
DEMO_STEPS = 3000
# =======================================

ROBOT = "Walker2d"
ENV_ID = "Walker2d-v4"
NOMBRE = f"{ROBOT}_FSSA_PPO{'_gSDE' if USE_GSDE else ''}"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELO_PATH = f"{RESULTS_DIR}/{NOMBRE}"

print(f"🎬 Cargando {MODELO_PATH}...")

# Entorno limpio (sin wrapper ni modulator) para evaluación pura
env = gym.make(ENV_ID, render_mode="human")
modelo = PPO.load(MODELO_PATH)

obs, info = env.reset()
for _ in range(DEMO_STEPS):
    action, _ = modelo.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("✅ Demo terminada")