import sys
import os
import time
import numpy as np
import gymnasium as gym

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fssa.wrapper import FaultCaptureWrapper

# ===== CONFIGURACIÓN =====
ROBOTS = [
    ("Walker2d", "Walker2d-v4"),
    ("Ant",      "Ant-v4"),
    ("Humanoid", "Humanoid-v4"),
]
N_PASOS = 5000   # pasos a medir por robot
# =========================

print("🧪 Test de overhead computacional del wrapper")
print("=" * 60)
print(f"  Pasos por medición: {N_PASOS}")
print()

resultados = []

for robot, env_id in ROBOTS:
    print(f"▶️  {robot}")

    # ── Medición SIN wrapper ──
    env = gym.make(env_id)
    obs, info = env.reset()

    t0 = time.perf_counter()
    for _ in range(N_PASOS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    t_sin = time.perf_counter() - t0
    env.close()

    # ── Medición CON wrapper ──
    env = gym.make(env_id)
    env = FaultCaptureWrapper(env)
    obs, info = env.reset()

    t0 = time.perf_counter()
    for _ in range(N_PASOS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    t_con = time.perf_counter() - t0
    env.close()

    # Cálculo
    ms_sin = (t_sin / N_PASOS) * 1000
    ms_con = (t_con / N_PASOS) * 1000
    overhead_ms = ms_con - ms_sin
    overhead_pct = (overhead_ms / ms_sin) * 100

    print(f"  Sin wrapper: {ms_sin:.4f} ms/paso ({t_sin:.2f} s totales)")
    print(f"  Con wrapper: {ms_con:.4f} ms/paso ({t_con:.2f} s totales)")
    print(f"  Overhead:    {overhead_ms:+.4f} ms/paso ({overhead_pct:+.2f}%)")
    print()

    resultados.append({
        "robot": robot,
        "ms_sin": ms_sin,
        "ms_con": ms_con,
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
    })

# Resumen
print("=" * 60)
print("📊 RESUMEN")
print("=" * 60)
print(f"  {'Robot':<10} | {'Sin (ms)':>10} | {'Con (ms)':>10} | {'Overhead':>12}")
print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
for r in resultados:
    print(f"  {r['robot']:<10} | {r['ms_sin']:>10.4f} | {r['ms_con']:>10.4f} | "
          f"{r['overhead_pct']:>+11.2f}%")

print()
print("✅ Test completado")