import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import matplotlib.pyplot as plt
import numpy as np
import os

# ===== CONFIGURACIÓN - CAMBIA AQUÍ =====
TOTAL_STEPS = 1000_000   # ← sube a 500_000 o 1_000_000 para mejor resultado
VER_DEMO = True         # ← False si no quieres ver la ventana al final
DEMO_STEPS = 1000       # ← cuántos pasos dura la demo visual
MODELO_PATH = "walker_ppo"  # ← nombre del archivo donde se guarda
# =======================================

# Callback para guardar recompensas y graficar
class GraficaCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.recompensas = []
        self.episodios = []
        self.ep_actual = 0
        self.recompensa_acum = 0

    def _on_step(self):
        self.recompensa_acum += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.ep_actual += 1
            self.recompensas.append(self.recompensa_acum)
            self.episodios.append(self.ep_actual)
            print(f"Episodio {self.ep_actual} | Recompensa: {self.recompensa_acum:.1f} | Pasos: {self.num_timesteps}")
            self.recompensa_acum = 0
        return True

    def graficar(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episodios, self.recompensas, alpha=0.4, color="steelblue", label="Recompensa")
        # Media móvil
        if len(self.recompensas) >= 10:
            media = np.convolve(self.recompensas, np.ones(10)/10, mode="valid")
            plt.plot(self.episodios[9:], media, color="red", linewidth=2, label="Media móvil (10 ep)")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa total")
        plt.title("Entrenamiento Walker2d-v4 con PPO")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("grafica_entrenamiento.png")
        print("✅ Gráfica guardada en grafica_entrenamiento.png")
        plt.show()

# ── Entrenamiento ──
print("🚀 Iniciando entrenamiento...")
env = gym.make("Walker2d-v4")
callback = GraficaCallback()

modelo = PPO("MlpPolicy", env, verbose=0)
modelo.learn(total_timesteps=TOTAL_STEPS, callback=callback)
modelo.save(MODELO_PATH)
print(f"✅ Modelo guardado en {MODELO_PATH}.zip")

env.close()
callback.graficar()

# ── Demo visual ──
if VER_DEMO:
    input("\n¿Ver demo con el modelo entrenado? Presiona Enter para continuar...")
    print("🎬 Abriendo demo...")
    env_demo = gym.make("Walker2d-v4", render_mode="human")
    modelo_cargado = PPO.load(MODELO_PATH)
    obs, info = env_demo.reset()

    for _ in range(DEMO_STEPS):
        action, _ = modelo_cargado.predict(obs)
        obs, reward, terminated, truncated, info = env_demo.step(action)
        if terminated or truncated:
            obs, info = env_demo.reset()

    env_demo.close()
    print("✅ Demo terminada")