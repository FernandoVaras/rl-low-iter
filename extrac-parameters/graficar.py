# Cargar datos
df = pd.read_csv("mujoco_data.csv")

# Altura del torso
plt.figure()
plt.plot(df["time"], df["torso_height"])
plt.xlabel("Tiempo")
plt.ylabel("Altura (m)")
plt.title("Altura del torso")
plt.show()

# Velocidad
plt.figure()
plt.plot(df["time"], df["velocity_x"])
plt.xlabel("Tiempo")
plt.ylabel("Velocidad X (m/s)")
plt.title("Velocidad de avance")
plt.show()

# Energía
plt.figure()
plt.plot(df["time"], df["energy"])
plt.xlabel("Tiempo")
plt.ylabel("Energía (J)")
plt.title("Energía total")
plt.show()

# Centro de masa (trayectoria)
plt.figure()
plt.plot(df["com_x"], df["com_z"])
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Trayectoria del CoM")
plt.show()
