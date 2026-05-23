# Pieza 1+2+3 — Vector Culpa

## ¿Qué hace?

Recibe la ventana pre-fallo del wrapper y devuelve un vector `c ∈ [0,1]^na`
que indica qué junta fue más responsable del fallo, agregado sobre todos
los timesteps de la ventana.

- **Entrada:** ventana del wrapper (8 señales × N timesteps)
- **Salida:** vector `c` shape (na,) con la junta más culpable en `1.0`

---

## Diseño actual: anomalía multi-señal

c_i = w1 · |qvel_i^norm| + w2 · |Δτ_i^norm|

Donde:
- `qvel_i^norm` = velocidad articular normalizada por su máximo de ventana
- `Δτ_i = ctrl_i - actuator_force_i` (delta torque)
- `|Δτ_i^norm|` = delta torque normalizado por su máximo de ventana
- Pesos default: `w1 = 0.4`, `w2 = 0.6`

### Justificación de los pesos

- `|q̇|` alta puede ser movimiento legítimo (ej. pierna en swing), solo es
  señal de fallo si es anormalmente alta
- `|Δτ|` alto es señal más limpia: significa que el controlador está pidiendo
  algo que el motor no puede dar (saturación inminente, fuerzas externas inesperadas)
- En la literatura de fault detection en robótica, el error entre comando
  y respuesta del actuador es el indicador más usado

### Subpasos del algoritmo

1. **Filtrar señales**
   - `qvel` viene en espacio nv → filtrar con `actuated_indices` a na
   - `ctrl` y `actuator_force` ya están en espacio na → no filtrar
2. **Normalización por junta** — cada junta normalizada por su propio máximo
   de ventana, así juntas con distinta escala (cadera vs tobillo) son comparables
3. **Combinar señales ponderadas** → `c_t = w1·qvel_norm + w2·delta_tau_norm`
4. **Agregación temporal exponencial** — pesos `w_t = exp(-λ·(N-1-t))` con λ=0.1,
   timesteps cercanos al fallo pesan más
5. **Normalización final** — dividir por el máximo, la junta más culpable queda en 1.0

---

## Diseño descartado: Jacobiano del torso

### Idea original

La fórmula original era:

c_t = pinv(J_torso) @ ẋ_torso

Donde `J_torso` es el Jacobiano espacial del torso (6×nv) y `ẋ_torso` la
velocidad espacial del cuerpo crítico. Esta fórmula funciona bien para
**manipuladores con base fija** (Franka, UR5), donde el end-effector es
el cuerpo crítico y todas las juntas actuadas contribuyen a su posición.

### Por qué no funciona en locomoción

En MuJoCo el `torso` siempre es la **raíz del modelo** para robots de
locomoción (Walker2d, Ant, Humanoid). Los 3 primeros DOFs (rootx, rootz, rooty)
son la junta libre del torso flotando. Las juntas actuadas mueven las piernas
**respecto al torso**, no al torso respecto al mundo.

Matemáticamente:
- Las columnas `J[:, j]` para juntas actuadas son ~0 (girar una rodilla no
  mueve el torso por sí solo en estática)
- `pinv(J)` asigna toda la culpa a la raíz porque es la única forma de
  generar ese `ẋ_torso`
- Al filtrar con `actuated_indices`, el vector resultante queda en ceros

### Evidencia del bug

Test end-to-end con modelo Walker2d_PPO entrenado, ventana de 50 pasos pre-fallo:

v_lin (cvel[torso][3:]):  [3.17, 0, -1.47]
v_ang (cvel[torso][:3]):  [0, 1.04, 0]
x_dot_torso:              [3.17, 0, -1.47, 0, 1.04, 0]
c_t_full:                 [3.67, -2.35, -0.69, 0, 0, 0, 0, 0, 0]
↑─── raíz ───↑  ↑─── actuadas ───↑

Toda la culpa cayó en los DOFs 0-2 (raíz). Al filtrar con
`actuated_indices=[3,4,5,6,7,8]`, el vector quedó en ceros.

### Caminos explorados antes de decidir

1. **Jacobiano de los pies** — fiel a la idea original pero requiere definir
   "movimiento anómalo del pie", lo cual no es trivial (los pies se mueven
   siempre durante la marcha)
2. **Jacobiano contact-aware** con `cfrc_ext` — físicamente correcto pero
   matemáticamente complejo, requiere identificar puntos de contacto activos
3. **Restar contribución de la raíz** — matemáticamente no funciona en estática,
   `J_actuado · q̇_actuado ≈ 0` para el torso
4. **Anomalía multi-señal** (elegido) — abandona el Jacobiano para Pieza 1

### Justificación para el paper

> "En tareas donde el cuerpo crítico está fuera de la cadena raíz
> (manipuladores), el Jacobiano del end-effector ofrece atribución
> cinemática directa. En locomoción, donde el cuerpo crítico ES la raíz,
> recurrimos a atribución basada en anomalía de señales articulares."

Esto convierte la limitación en una decisión de diseño justificada.

---

## Resultado del test end-to-end

Walker2d con PPO entrenado, ventana de 50 pasos antes de la caída:

Vector c: [0.2481, 0.7234, 0.1888, 1.0000, 0.5051, 0.0398]
Junta más culpable: índice 3 (cadera izquierda)

Interpretación física: la cadera izquierda, seguida de las rodillas derecha
e izquierda, fueron las articulaciones con mayor anomalía antes del fallo.
Coherente con el rol de las articulaciones grandes en el equilibrio.

---

## Hiperparámetros (TODO: mover a config YAML)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `LAMBDA_TEMPORAL` | 0.1 | Decaimiento exponencial temporal |
| `W_QVEL` | 0.4 | Peso de anomalía de velocidad articular |
| `W_DELTA_TAU` | 0.6 | Peso de anomalía de delta torque |
| `EPSILON` | 1e-8 | Evita división por cero en normalización |

### Ablaciones sugeridas para el paper

Reportar resultados con distintas combinaciones de pesos:
- `(0.5, 0.5)` — igual peso
- `(0.7, 0.3)` — favorece qvel
- `(0.3, 0.7)` — favorece delta torque
- `(0.4, 0.6)` — default actual