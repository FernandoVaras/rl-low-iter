import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MODELOS = {
    "PPO":      f"{RESULTS_DIR}/Ant_PPO_entrenamiento.csv",
    "PPO gSDE": f"{RESULTS_DIR}/Ant_PPO_gSDE_entrenamiento.csv",
    "SAC":      f"{RESULTS_DIR}/Ant_SAC_entrenamiento.csv",
    "SAC gSDE": f"{RESULTS_DIR}/Ant_SAC_gSDE_entrenamiento.csv",
}

COLORES = {
    "PPO":      "#2196F3",
    "PPO gSDE": "#4CAF50",
    "SAC":      "#F44336",
    "SAC gSDE": "#FF9800",
}

datos = {}
for nombre, path in MODELOS.items():
    if os.path.exists(path):
        datos[nombre] = pd.read_csv(path)
        print(f"✅ Cargado: {nombre}")
    else:
        print(f"⚠️  No encontrado: {nombre} ({path})")

if not datos:
    print("❌ No hay CSVs disponibles en results/")
    exit()

# ── Gráfica 1: Episodio vs Recompensa ──
max_episodios = max(df["episodio"].max() for df in datos.values())

plt.figure(figsize=(12, 6))
for nombre, df in datos.items():
    plt.plot(df["episodio"], df["recompensa"],
             alpha=0.3, color=COLORES[nombre])
    if len(df) >= 10:
        media = np.convolve(df["recompensa"], np.ones(10)/10, mode="valid")
        plt.plot(df["episodio"].iloc[9:], media,
                 color=COLORES[nombre], linewidth=2, label=nombre)
    else:
        plt.plot(df["episodio"], df["recompensa"],
                 color=COLORES[nombre], linewidth=2, label=nombre)

plt.xlim(0, max_episodios)
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.title("Ant — Recompensa por Episodio")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/Ant_episode_rewards.png")
print("✅ Guardado: Ant_episode_rewards.png")
plt.close()

# ── Gráfica 2: Pasos vs Recompensa ──
max_pasos = max(df["pasos"].max() for df in datos.values())

plt.figure(figsize=(12, 6))
for nombre, df in datos.items():
    pasos_comun = np.linspace(0, max_pasos, 1000)
    recompensa_interp = np.interp(pasos_comun, df["pasos"], df["recompensa"])

    plt.plot(pasos_comun, recompensa_interp,
             alpha=0.3, color=COLORES[nombre])
    media = np.convolve(recompensa_interp, np.ones(50)/50, mode="valid")
    plt.plot(pasos_comun[49:], media,
             color=COLORES[nombre], linewidth=2, label=nombre)

plt.xlim(0, max_pasos)
plt.xlabel("Pasos")
plt.ylabel("Recompensa total")
plt.title("Ant — Recompensa por Pasos")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/Ant_rewards_step.png")
print("✅ Guardado: Ant_rewards_step.png")
plt.close()