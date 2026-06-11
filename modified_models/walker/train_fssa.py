import sys
import os
import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fssa.wrapper import FaultCaptureWrapper
from fssa.module import FSSAModule
from fssa.modulator import FSSAModulator
from fssa.utils import load_config

# ===== CONFIGURACIÓN - CAMBIA AQUÍ =====
USE_GSDE = False
TOTAL_STEPS = 700_000
config = load_config("walker2d")
# =======================================

ROBOT = "Walker2d"
ENV_ID = "Walker2d-v4"
ALGORITMO = "PPO"
NOMBRE = f"{ROBOT}_FSSA_{ALGORITMO}{'_gSDE' if USE_GSDE else ''}"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELO_PATH = f"{RESULTS_DIR}/{NOMBRE}.zip"
PROGRESO_PATH = f"{RESULTS_DIR}/{NOMBRE}_pasos.txt"
GRAFICA_PATH = f"{RESULTS_DIR}/grafica_{NOMBRE}.png"
CSV_PATH = f"{RESULTS_DIR}/{NOMBRE}_entrenamiento.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)


class FSSACallback(BaseCallback):
    """
    Callback que conecta el FSSAModule al entrenamiento de SB3.

    Responsabilidades:
    - Inicio de cada episodio → fssa.step_episode() para decaer g
    - Cuando hay fallo → fssa.process_fault(window)
    - Registrar recompensas para gráfica/CSV
    """

    def __init__(self, fssa_module):
        super().__init__()
        self.fssa = fssa_module
        self.recompensas = []
        self.episodios = []
        self.pasos = []
        self.ep_actual = 0
        self.recompensa_acum = 0
        self.episodio_recien_iniciado = True

    def _on_step(self):
        # Acumular recompensa del episodio actual
        self.recompensa_acum += self.locals["rewards"][0]

        # Aplicar decaimiento al inicio de cada episodio
        if self.episodio_recien_iniciado:
            self.fssa.step_episode()
            self.episodio_recien_iniciado = False

        # Procesar fallo si lo hay
        info = self.locals["infos"][0]
        if "fault_window" in info:
            self.fssa.process_fault(info["fault_window"])

        # Si terminó el episodio
        if self.locals["dones"][0]:
            self.ep_actual += 1
            self.recompensas.append(self.recompensa_acum)
            self.episodios.append(self.ep_actual)
            self.pasos.append(self.num_timesteps)
            print(f"Episodio {self.ep_actual} | Recompensa: {self.recompensa_acum:.1f} | Pasos: {self.num_timesteps}")
            self.recompensa_acum = 0
            self.episodio_recien_iniciado = True

        return True

    def graficar(self):
        if len(self.recompensas) < 2:
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.episodios, self.recompensas, alpha=0.4, color="steelblue", label="Recompensa")
        if len(self.recompensas) >= 10:
            media = np.convolve(self.recompensas, np.ones(10)/10, mode="valid")
            plt.plot(self.episodios[9:], media, color="red", linewidth=2, label="Media móvil (10 ep)")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa total")
        plt.title(f"Entrenamiento {NOMBRE}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(GRAFICA_PATH)
        print(f"✅ Gráfica guardada en {GRAFICA_PATH}")

    def guardar_csv(self):
        if len(self.recompensas) < 1:
            return
        df = pd.DataFrame({
            "episodio": self.episodios,
            "pasos": self.pasos,
            "recompensa": self.recompensas
        })
        df.to_csv(CSV_PATH, index=False)
        print(f"✅ CSV guardado en {CSV_PATH}")


# Cargar pasos previos
pasos_previos = 0
if os.path.exists(PROGRESO_PATH):
    with open(PROGRESO_PATH, "r") as f:
        pasos_previos = int(f.read().strip())
    print(f"📂 Continuando desde {pasos_previos} pasos...")
else:
    print(f"🚀 Entrenamiento FSSA nuevo: {NOMBRE}")

pasos_restantes = TOTAL_STEPS - pasos_previos
if pasos_restantes <= 0:
    print("✅ Ya completaste el entrenamiento total!")
    sys.exit()

# Crear pila: env → FaultCaptureWrapper → FSSAModulator
env = gym.make(ENV_ID)
env = FaultCaptureWrapper(env, config=config)
fssa = FSSAModule(env, config=config)
env = FSSAModulator(env, fssa, config=config)

callback = FSSACallback(fssa)

if os.path.exists(MODELO_PATH):
    modelo = PPO.load(MODELO_PATH, env=env)
else:
    modelo = PPO("MlpPolicy", env, verbose=0, use_sde=USE_GSDE)


def guardar_y_salir(sig, frame):
    print("\n⚠️ Interrumpido! Guardando...")
    modelo.save(MODELO_PATH)
    pasos_totales = pasos_previos + callback.num_timesteps
    with open(PROGRESO_PATH, "w") as f:
        f.write(str(pasos_totales))
    print(f"✅ Guardado en {MODELO_PATH} ({pasos_totales} pasos totales)")
    callback.graficar()
    callback.guardar_csv()
    sys.exit(0)


signal.signal(signal.SIGINT, guardar_y_salir)

try:
    modelo.learn(total_timesteps=pasos_restantes, callback=callback, reset_num_timesteps=False)
    modelo.save(MODELO_PATH)
    with open(PROGRESO_PATH, "w") as f:
        f.write(str(TOTAL_STEPS))
    print(f"✅ Entrenamiento FSSA completo! {TOTAL_STEPS} pasos totales")
    callback.graficar()
    callback.guardar_csv()
except Exception as e:
    print(f"Error: {e}")
    guardar_y_salir(None, None)

env.close()