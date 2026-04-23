import gymnasium as gym
from stable_baselines3 import PPO

MODELO_PATH = "walker_ppo"  # ← nombre de tu modelo guardado
DEMO_STEPS = 20000           # ← cuántos pasos dura la demo

env = gym.make("Walker2d-v4", render_mode="human")
modelo = PPO.load(MODELO_PATH)
obs, info = env.reset()

for _ in range(DEMO_STEPS):
    action, _ = modelo.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("✅ Demo terminada")