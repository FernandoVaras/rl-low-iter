# mj_jac — Jacobiano en MuJoCo

## ¿Qué es?
Es una función de MuJoCo que calcula el **Jacobiano** de un punto específico
del cuerpo del robot en un instante congelado de la simulación.

No mueve nada, solo analiza la postura actual y calcula matemáticamente:
*"En esta postura exacta, si aplicara un movimiento infinitesimal a cada
articulación por separado, ¿cuánto se movería este punto?"*

---

## Entradas

```python
mujoco.mj_jac(model, data, jacp, jacr, punto, body_id)
```

| Parámetro | Qué es | Ejemplo |
|-----------|--------|---------|
| `model` | el robot cargado | `mujoco.MjModel.from_xml_path(...)` |
| `data` | estado actual de la simulación | `mujoco.MjData(model)` |
| `jacp` | matriz vacía 3×nv donde escribe el resultado traslacional | `np.zeros((3, model.nv))` |
| `jacr` | matriz vacía 3×nv donde escribe el resultado rotacional | `np.zeros((3, model.nv))` |
| `punto` | posición actual del cuerpo en el espacio (XYZ) | `data.xpos[body_id]` |
| `body_id` | qué parte del cuerpo analizar | pie derecho, torso, rodilla... |

### ¿Por qué el punto en el espacio?
El Jacobiano no se calcula para el cuerpo entero sino para un **punto específico
dentro de ese cuerpo**. Imagina el pie del Walker, ¿calculas para la punta,
el talón o el centro? Cada punto da resultado diferente.

Lo más común es usar el centro del cuerpo que MuJoCo ya conoce:
```python
data.xpos[body_id]  # posición actual del centro del body en el mundo
```
No es 0,0,0 — es la posición real. Si el pie está en x=1.5, z=0.1
entonces el punto es [1.5, 0.0, 0.1] y cambia en cada paso.

---

## Salidas

- `jacp` → matriz **3 × nv**
  - fila 1 = cuánto se mueve el punto en **X** por cada articulación
  - fila 2 = cuánto se mueve en **Y**
  - fila 3 = cuánto se mueve en **Z**
- `jacr` → igual pero para **rotación**

### Ejemplo de cómo se lee la matriz (Walker2d tiene 6 articulaciones)
         cadera   rodilla   tobillo   cadera2   rodilla2   tobillo2
X →        [ 0.8      0.5       0.2       0.0       0.0        0.0     ]
Y →        [ 0.0      0.0       0.0       0.0       0.0        0.0     ]
Z →        [-0.3     -0.4      -0.1       0.0       0.0        0.0     ]

Lo que dice cada columna:
- **cadera derecha** → si la muevo, el pie derecho se desplaza 0.8 en X y -0.3 en Z
- **rodilla derecha** → si la muevo, el pie se desplaza 0.5 en X y -0.4 en Z
- **cadera izquierda** → casi 0, porque la pierna izquierda no afecta al pie derecho

Todo esto asumiendo que **todo lo demás está estático** en ese instante.
Es una derivada parcial: "solo esta articulación, todo lo demás quieto".

---

## Importante: es una foto, no un video

El Jacobiano se calcula en **un instante congelado**. La postura del robot
en ese momento determina el resultado. Por eso:

- Paso 1 (pie en el aire) → Jacobiano A
- Paso 2 (pie tocando suelo) → Jacobiano B
- Paso 3 (empujando) → Jacobiano C

Cada postura da un Jacobiano diferente.

---

## Uso básico

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("walker2d.xml")
data = mujoco.MjData(model)

# Obtener body_id del pie derecho
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_right")

# Matrices vacías donde se escribirá el resultado
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

# Avanzar un paso y calcular
mujoco.mj_step(model, data)
mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)

print(jacp)  # matriz 3x6 con el Jacobiano traslacional
```

---

## Jacobiano a través del tiempo

Como el Jacobiano depende de la postura, calcularlo en cada paso
te da cómo evoluciona mientras el robot camina:

```python
jacobianos = []

for i in range(steps):
    mujoco.mj_step(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)
    jacobianos.append(jacp.copy())  # guardar copia de cada instante

# jacobianos[0] = Jacobiano en paso 1
# jacobianos[1] = Jacobiano en paso 2
# ...
```

Resultado: una lista de matrices, una por cada instante = **Jacobiano en el tiempo**.

---

## ¿Para qué sirve en este proyecto?

Cuando el Walker camina normal el Jacobiano tiene un **patrón característico**.
Cuando hay un fallo (articulación dañada, sensor roto) ese patrón **cambia**.

Esa diferencia entre el patrón normal y el patrón con fallo es la
**firma sensorial del fallo** que este proyecto busca detectar.

Ejemplos:
- Rodilla derecha dañada → columna 2 de jacp cambia drásticamente
- Tobillo izquierdo bloqueado → columna 6 casi en cero
- Cadera débil → columna 1 con valores reducidos