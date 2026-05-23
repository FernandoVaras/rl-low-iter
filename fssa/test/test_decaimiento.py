import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fssa.pieces.decaimiento import apply_decay

# ===== CONFIGURACIÓN =====
G_INICIAL = np.array([1.0, 2.0, 1.0, 1.9, 2.0, 1.0])  # ejemplo Walker2d
EPISODIOS = 15
LAMBDA_EP = 0.3
# =========================

print("🧪 Test de decaimiento exponencial episódico")
print("=" * 60)
print(f"  g_inicial: {G_INICIAL}")
print(f"  lambda_ep: {LAMBDA_EP}")
print(f"  Episodios a simular: 0 a {EPISODIOS}")

print(f"\n📊 EVOLUCIÓN DE g:")
print(f"  {'k':>3} | {'g_actual'}")
print(f"  {'-'*3}-+-{'-'*40}")

historial = []
for k in range(EPISODIOS + 1):
    g_actual = apply_decay(G_INICIAL, k, LAMBDA_EP)
    historial.append({"k": k, **{f"junta_{i}": v for i, v in enumerate(g_actual)}})
    print(f"  {k:>3} | {np.round(g_actual, 4)}")

# Validación
print(f"\n✅ VALIDACIÓN:")
g_final = apply_decay(G_INICIAL, EPISODIOS, LAMBDA_EP)
print(f"  g en k={EPISODIOS}: {np.round(g_final, 4)}")
print(f"  Todas las juntas cerca de 1.0: {np.all(np.abs(g_final - 1.0) < 0.05)}")
print(f"  g siempre >= 1.0: {np.all(g_final >= 1.0)}")

# Guardar CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), "decaimiento_evolucion.csv")
df = pd.DataFrame(historial)
df.to_csv(CSV_PATH, index=False)
print(f"\n💾 Guardado en {CSV_PATH}")

print("\n" + "=" * 60)
print("✅ Test completado")