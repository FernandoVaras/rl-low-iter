# Pieza 4 — Vector Margen

## ¿Qué hace?

Recibe la ventana pre-fallo del wrapper y devuelve un vector `m ∈ [0,1]^na`
que indica qué tan cerca está cada motor de saturar su torque máximo.

- **Entrada:** ventana del wrapper (señal `actuator_force`) + `tau_max` del modelo
- **Salida:** vector `m` shape (na,) donde:
  - `1.0` = motor totalmente libre (no usa nada de su capacidad)
  - `0.0` = motor saturado (al 100% de su capacidad)

---

## Fórmula

m_i^t = 1 - |τ_i^t| / τ_max_i
m_aggregated = sum_t (w_t · m_i^t)
m_clipped = clip(m_aggregated, 0, 1)
Donde:
- `τ_i^t` = `actuator_force` real aplicado por el motor i en el timestep t
- `τ_max_i` = torque máximo absoluto del motor (de `utils/mujoco_helpers.get_tau_max`)
- `w_t` = pesos exponenciales temporales (mismos que culpa, λ=0.1)

### Subpasos del algoritmo

1. **Cálculo de saturación instantánea** por timestep y motor
2. **Margen instantáneo** = 1 - saturación
3. **Agregación temporal exponencial** — timesteps cercanos al fallo pesan más
4. **Clip a [0, 1]** por seguridad numérica

---

## Interpretación física

| Margen | Significado |
|--------|-------------|
| `1.00` | Motor sin usar, máxima reserva |
| `0.99` | Usa 1% de su capacidad |
| `0.50` | Medio camino al límite |
| `0.10` | Al borde de saturar, sin reserva |
| `0.00` | Saturado, no puede dar más |

Un motor con margen bajo es candidato a fallar porque no puede compensar más demanda.
Un motor con margen alto tiene reserva para absorber perturbaciones.

---

## Cálculo de tau_max

El torque máximo se obtiene del modelo MuJoCo con la siguiente prioridad:

1. **Si `actuator_forcelimited == True`** → usar `max(|actuator_forcerange|)`
2. **Si no** → usar `gear × max(|ctrlrange|)` como fallback

El fallback es necesario porque los XMLs de Gymnasium (Walker2d, Humanoid)
no definen `actuator_forcerange` explícito, pero sí definen `gear` y `ctrlrange`,
cuya multiplicación da el torque efectivo del motor.

Ejemplo Walker2d:
```xml
<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" .../>
```
→ `tau_max = 1.0 × 100 = 100 Nm`

Ejemplo Humanoid:

tau_max = [40, 40, 40, 40, 40, 120, 80, 40, 40, 120, 80, 10, 10, 10, 10, 10, 10]
↑─── torso/abdomen ───↑  ↑─cadera↑ ↑─rodilla↑  ↑──── brazos ────↑

---

## Resultados del test end-to-end

### Walker2d (PPO entrenado, 700k pasos)

tau_max:  [100, 100, 100, 100, 100, 100]
Vector m: [0.9907, 0.9943, 0.9911, 0.9949, 0.9933, 0.9904]

Todos los motores con margen ~0.99. **El margen no discrimina bien en Walker2d**
porque el robot falla por inestabilidad postural (caída) y no llega a saturar
motores antes del fallo. Los 100 Nm de capacidad por motor son excesivos para
caminar normal.

### Humanoid (PPO entrenado, 700k pasos)

tau_max:  [40, 40, 40, 40, 40, 120, 80, 40, 40, 120, 80, 10, 10, 10, 10, 10, 10]
Vector m: [0.9924, 0.9913, 0.9905, 0.9914, 0.9902, 0.9970, 0.9954, 0.9916,
0.9915, 0.9970, 0.9950, 0.9642, 0.9676, 0.9708, 0.9687, 0.9652, 0.9633]
↑──────────── piernas/torso (margen ~0.99) ────────────↑
↑──── brazos (margen ~0.96) ────↑

Los **brazos** (motores 11-16) tienen márgenes más bajos porque su `tau_max=10`
es pequeño. Aunque hagan trabajo ligero, están porcentualmente más cerca de
su límite que las piernas que tienen `tau_max=80-120`.

### Humanoid (SAC entrenado, 700k pasos)

Vector m: [0.9917, 0.9918, 0.9921, 0.9930, 0.9934, 0.9974, 0.9957, 0.9920,
0.9932, 0.9974, 0.9958, 0.9719, 0.9773, 0.9749, 0.9755, 0.9692, 0.9709]

SAC tiene márgenes ligeramente más altos en brazos que PPO (0.97 vs 0.96),
confirmando que SAC camina más eficiente que PPO. Coherente con la diferencia
de recompensa observada en las curvas de entrenamiento.

---

## Observaciones para el paper

1. **El margen es más informativo en robots complejos (Humanoid) que en simples (Walker2d).**
   En Walker los fallos son por inestabilidad, no saturación. En Humanoid el margen
   captura qué motores trabajan cerca de su límite físico real.

2. **El margen no detecta fallos por sí solo en estos benchmarks**, pero aporta
   información complementaria a culpa: cuáles motores aún tienen capacidad
   para compensar y cuáles no.

3. **La interpretación es absoluta**, no relativa entre motores. `m=0.5` siempre
   significa "50% saturado", independientemente del robot.

---

## Hiperparámetros (TODO: mover a config YAML)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `LAMBDA_TEMPORAL` | 0.1 | Decaimiento exponencial (mismo que culpa) |
| `EPSILON` | 1e-8 | Evita división por cero si tau_max=0 |