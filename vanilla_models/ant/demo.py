import gymnasium as gym
from stable_baselines3 import PPO, SAC
import os

# ===== CONFIGURACIÓN - CAMBIA AQUÍ =====
ALGORITMO = "SAC"       # ← PPO, SAC
USE_GSDE = False        # ← igual que en train.py
DEMO_STEPS = 3000
# =======================================

ROBOT = "Ant"
ENV_ID = "Ant-v4"
NOMBRE = f"{ROBOT}_{ALGORITMO}{'_gSDE' if USE_GSDE else ''}"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELO_PATH = f"{RESULTS_DIR}/{NOMBRE}.zip"

print(f"🎬 Cargando {MODELO_PATH}...")

env = gym.make(ENV_ID, render_mode="human")

if ALGORITMO == "PPO":
    modelo = PPO.load(MODELO_PATH)
elif ALGORITMO == "SAC":
    modelo = SAC.load(MODELO_PATH)

obs, info = env.reset()
for _ in range(DEMO_STEPS):
    action, _ = modelo.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("✅ Demo terminada")