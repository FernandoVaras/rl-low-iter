# Pieza 5 — Combinación (Culpa × Margen)

## ¿Qué hace?

Combina el vector culpa y el vector margen en un único factor de modulación
`g ∈ [g_min, g_max]^na` que indica cuánto amplificar la exploración por junta
en el siguiente episodio.

- **Entrada:** vector culpa `c` y vector margen `m`
- **Salida:** vector `g` shape (na,) en [1.0, 2.0]
  - `g_i = 1.0` → sin modulación (exploración base)
  - `g_i = 2.0` → modulación máxima (amplificar exploración)

---

## Fórmula

### Sub-paso 1: combinación multiplicativa con exponentes
g_raw_i = (c_i + ε)^α · (m_i + ε)^(1-α)

Con `α = 0.7` → más peso a culpa que a margen.

El `ε = 1e-8` evita el caso indefinido `0^0` en numpy si culpa o margen son
exactamente cero.

### Sub-paso 2: re-escalado centrado en la media
g_final_i = 1 + κ · (g_raw_i - mean(g_raw))

Con `κ = 4.0` → amplificación agresiva alrededor de la media.

Esto convierte el vector `g_raw` en un vector centrado en 1.0 donde:
- Juntas con `g_raw` por encima de la media → `g_final > 1` (amplificar)
- Juntas con `g_raw` por debajo de la media → `g_final < 1` (reducir)

### Sub-paso 3: clip asimétrico
g_clipped = clip(g_final, g_min, g_max)

Con `g_min = 1.0` y `g_max = 2.0`.

**Importante:** el piso es `1.0`, no `0.5`. Esto significa que el módulo
**solo amplifica, nunca reduce** la exploración base. Si una junta no es
culpable, se queda en exploración base, no se penaliza.

---

## Por qué multiplicativa y no aditiva

La multiplicación implementa un AND lógico: una junta solo se modula fuerte
si **es culpable Y tiene margen** para compensar.

Casos extremos:
- Culpa alta + margen alto → `g` alto (sí modular)
- Culpa alta + margen bajo → `g` medio (modular menos, motor ya saturado)
- Culpa baja + margen alto → `g` bajo (no es responsable)
- Culpa baja + margen bajo → `g` bajo (no es responsable)

La opción aditiva (`α·c + β·m`) suma contribuciones independientes y pierde
esa interacción.

---

## Por qué α = 0.7

La culpa pesa más que el margen porque:
- Culpa identifica directamente al responsable del fallo
- Margen es información complementaria sobre capacidad de compensación
- Si una junta no es culpable, su margen no debería amplificar exploración

El 0.7 vs 0.3 es un balance: ~70% culpa, ~30% margen. Es ajustable como
ablación en el paper.

---

## Por qué κ = 4 y g_min = 1.0

### El κ controla qué tan agresiva es la modulación

Con `κ = 4`, una junta con `g_raw` ligeramente por encima de la media salta
rápido a `g = 2.0` (saturación máxima). Eso favorece a las juntas más
culpables y deja las demás cerca de 1.0.

### g_min = 1.0 es asimétrico intencionalmente

Originalmente el plan era `g_min = 0.5` (reducir exploración en juntas sanas).
El asesor lo cambió a `g_min = 1.0` para que el módulo **nunca reduzca la
exploración**. Razón: si fallamos la atribución (falso positivo en culpa),
reducir exploración en una junta sana puede romper el aprendizaje.

Es más seguro "no hacer nada" en juntas no culpables que "penalizar incorrectamente".

---

## Resultado del test end-to-end

Walker2d con PPO entrenado:
Vector c:  [0.3581, 0.8557, 0.1933, 1.0000, 0.9995, 0.1650]
Vector m:  [0.9929, 0.9939, 0.9920, 0.9938, 0.9926, 0.9907]
Vector g:  [1.0000, 1.9299, 1.0000, 2.0000, 2.0000, 1.0000]

Observación: como el margen es prácticamente uniforme (~0.99), la fórmula
efectivamente se reduce a "modular según culpa". Las juntas con culpa baja
quedan en `g=1.0` (sin modulación) y las culpables saturan a `g=2.0`.

Esto es coherente con lo discutido en la documentación de margen:
en Walker2d el margen no aporta discriminación, la culpa hace el trabajo.

---

## Hiperparámetros (TODO: mover a config YAML)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `ALPHA` | 0.7 | Exponente de culpa (1-α = exponente de margen) |
| `KAPPA` | 4.0 | Factor de amplificación alrededor de la media |
| `G_MIN` | 1.0 | Piso del clip — solo amplifica, no reduce |
| `G_MAX` | 2.0 | Techo del clip |
| `EPSILON` | 1e-8 | Evita 0^0 indefinido |

### Ablaciones sugeridas para el paper

- α: `0.5, 0.7, 0.9` — peso de culpa vs margen
- κ: `2, 4, 8` — agresividad de la amplificación
- g_max: `1.5, 2.0, 3.0` — techo de modulación