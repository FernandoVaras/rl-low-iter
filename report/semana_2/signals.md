# Señales disponibles en MuJoCo — Walker2d

> Extraído usando `inspect_model.py` sobre `walker2d.xml`

---

## Todas las señales disponibles

### Por joint (shape 9,) — 9 joints en total
| Señal | Descripción |
|-------|-------------|
| `qpos` | Posición angular de cada joint |
| `qvel` | Velocidad angular de cada joint |
| `qacc` | Aceleración angular de cada joint |
| `qacc_smooth` | Aceleración suavizada |
| `qacc_warmstart` | Aceleración inicial del solver |
| `qfrc_actuator` | Fuerza del actuador por joint |
| `qfrc_constraint` | Fuerza de restricción por joint |
| `qfrc_bias` | Fuerzas de Coriolis y gravedad |
| `qfrc_applied` | Fuerza externa aplicada manualmente |
| `qfrc_passive` | Fuerzas pasivas (amortiguamiento) |
| `qfrc_damper` | Fuerza de amortiguador |
| `qfrc_fluid` | Fuerza de fluido |
| `qfrc_gravcomp` | Compensación de gravedad |
| `qfrc_inverse` | Fuerza inversa dinámica |
| `qfrc_smooth` | Fuerza suavizada |
| `qfrc_spring` | Fuerza de resorte |
| `ifrc_constraint` | Fuerza de restricción interna |
| `ifrc_smooth` | Fuerza suavizada interna |
| `iacc` | Aceleración interna |
| `iacc_smooth` | Aceleración interna suavizada |

### Por actuador (shape 6,) — 6 actuadores en total
| Señal | Descripción |
|-------|-------------|
| `ctrl` | Señal de control enviada al actuador |
| `actuator_force` | Fuerza real aplicada por el actuador |
| `actuator_velocity` | Velocidad del actuador |
| `actuator_length` | Longitud del actuador |
| `actuator_moment` | Momento del actuador |

### Por body (shape 8,x) — 8 bodies en total
| Señal | Descripción |
|-------|-------------|
| `xpos` | Posición XYZ de cada body en el espacio |
| `xquat` | Orientación en quaternion (4 valores) |
| `xmat` | Orientación en matriz 3x3 (9 valores) |
| `xipos` | Posición del centro de inercia |
| `ximat` | Orientación del centro de inercia |
| `cvel` | Velocidad traslacional + rotacional (6 valores) |
| `cacc` | Aceleración traslacional + rotacional (6 valores) |
| `cfrc_ext` | Fuerzas externas sobre cada body (6 valores) |
| `cfrc_int` | Fuerzas internas sobre cada body (6 valores) |
| `cinert` | Inercia compuesta (10 valores) |
| `cdof` | Jacobiano de espacio de configuración (6 valores) |
| `cdof_dot` | Derivada del Jacobiano |
| `crb` | Cuerpo rígido compuesto |
| `subtree_com` | Centro de masa del subárbol XYZ |
| `subtree_linvel` | Velocidad lineal del subárbol XYZ |
| `subtree_angmom` | Momento angular del subárbol XYZ |
| `xfrc_applied` | Fuerza externa aplicada manualmente |
| `body_awake` | Si el body está activo (1/0) |
| `geom_xpos` | Posición XYZ de cada geometría |
| `geom_xmat` | Orientación de cada geometría |

### Global
| Señal | Descripción |
|-------|-------------|
| `energy` | Energía cinética y potencial (2 valores) |
| `time` | Tiempo de simulación |
| `ncon` | Número de contactos activos |

---

## Señales descartadas

Las siguientes señales están disponibles pero no son útiles para análisis
de comportamiento o detección de fallos:

- **Matrices internas del solver** (`efc_*`, `iM`, `qLD`, `qH`, etc.) →
  cálculos numéricos internos para resolver la física, no señales físicas reales
- **Mapeos e índices** (`map_*`, `island_*`, `dof_*`) →
  bookkeeping interno del solver
- **Señales vacías** (shape 0,) →
  tendon, flex, plugin, site, mocap — no aplican al Walker2d
- **Señales redundantes de orientación** (`xmat`, `ximat`, `geom_xmat`) →
  misma información que `xquat` pero en formato matricial

---

## Señales candidatas para el proyecto

> Filtro preliminar — refinado a set mínimo justificado

Reducir señales es importante porque cada señal extra añade ruido,
memoria y dificulta identificar qué causó qué. El principio es:
**mínimo suficiente con justificación clara**.

### Señales primarias (6)

| Señal | Por qué |
|-------|---------|
| `qpos` | Postura del robot, necesaria para evaluar el Jacobiano J(q) |
| `qvel` | Velocidad articular, necesaria para velocidad en el Jacobiano |
| `cfrc_ext` | Contacto con suelo, marca clara de pie despegado o caída |
| `subtree_com` (torso) | Indicador maestro de estabilidad, si se cae esto lo refleja primero |
| `actuator_force` | Torque real aplicado por cada motor |
| `ctrl` | Torque comandado al motor |

### Señales derivadas (4)

Calculadas a partir de las primarias, con sentido físico claro:

| Derivada | Cómo se calcula | Por qué |
|----------|----------------|---------|
| **Delta torque** | `ctrl - actuator_force` | Diferencia entre lo ordenado y lo ejecutado, detecta saturación por junta |
| **Inclinación del torso** | ángulo de `qpos[orientación]` | Señal más directa de "se está cayendo" |
| **Aceleración articular** | derivada temporal de `qvel` | Equivalente a `qacc` pero con más control sobre el filtrado |
| **Jerk del torso** | derivada de la aceleración de `subtree_com` | Detecta movimientos bruscos antes del fallo, anticipa la caída |

### Señales descartadas en este filtro

- `qacc` → se recalcula como derivada de `qvel`
- `actuator_velocity`, `actuator_length` → redundantes con `qvel`
- `actuator_force` y `qfrc_actuator` → casi idénticas, basta con una
- `qfrc_constraint`, `qfrc_bias` → útiles para análisis dinámico avanzado, no para este módulo
- `xpos` general → basta con `subtree_com` del torso
- `subtree_linvel`, `subtree_angmom` → informativas pero no críticas
- `energy`, `cvel` → derivables de las primarias, no aportan al mecanismo

**Total: 10 features por timestep.** Set manejable, justificable y suficiente.