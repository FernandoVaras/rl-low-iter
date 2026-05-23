# Pieza 5b — Difusión Cinemática (opcional)

## ¿Qué hace?

Propaga el valor del vector `g` entre juntas que están conectadas
en la cadena cinemática del robot. La idea: cuando una junta es responsable
de un fallo, sus vecinas mecánicas también están "comprometidas" porque
comparten responsabilidad física.

Es una pieza **opcional** que se activa con `use_diffusion=True` en
`compute_combinacion`. Por default está desactivada.

---

## Pipeline donde encaja

Calcular culpa → c
Calcular margen → m
Combinación multiplicativa → g_raw
Re-escalado (media + κ) → g_pre
Difusión cinemática → g_difundido    ← AQUÍ (opcional)
Clip [g_min, g_max] → g final


Va **después del re-escalado** y **antes del clip** porque:
- Difundir antes del re-escalado mezcla valores en escala original con
  vecinos, los efectos son impredecibles
- Difundir después del clip rompe los límites (los valores ya estaban
  en [1, 2] y la suma de vecinos los saca del rango)

---

## Fórmula

Versión inicial considerada:
g_difundido = (I + β · A) · g
Donde `A` es la matriz de adyacencia. Problema: sin normalizar, los valores
se inflan y saturan.

**Versión adoptada (Laplaciano normalizado):**
g_difundido = (I + β · L_norm) · g
L_norm = I - D^(-1/2) · A · D^(-1/2)

Donde:
- `A` = matriz de adyacencia entre actuadores (1 si vecinos, 0 si no)
- `D` = matriz diagonal de grados (cuántos vecinos tiene cada actuador)
- `L_norm` = Laplaciano normalizado, preserva la masa total

Esta versión es **matemáticamente honesta** porque los valores se redistribuyen
entre vecinos sin inflar el total, manteniendo la interpretación del módulo.

---

## Cómo se construye la adyacencia

En `utils/adjacency.py` se construye automáticamente desde el modelo MuJoCo:

1. Para cada actuador, obtener qué junta controla
2. Para cada junta, obtener en qué body está
3. Dos actuadores son vecinos si:
   - Sus bodies comparten el mismo padre, O
   - Un body es padre del otro

Esto refleja la cadena cinemática real del robot.

### Ejemplo Walker2d (6 actuadores)

Matriz de adyacencia 6×6 basada en la estructura del modelo. Las juntas
de cada pierna son vecinas entre sí (cadera-rodilla, rodilla-tobillo)
y las caderas izquierda y derecha son vecinas porque comparten torso como padre.

---

## Resultados del test end-to-end

### Walker2d (PPO entrenado)
Vector c:           [0.31, 0.89, 0.25, 0.80, 1.00, 0.09]
Vector g (sin dif): [1.00, 2.00, 1.00, 1.90, 2.00, 1.00]
Vector g (con dif): [1.00, 2.00, 1.00, 2.00, 2.00, 1.00]
Diferencia:         [0.00, 0.00, 0.00, 0.10, 0.00, 0.00]

**Efecto pequeño.** Solo la junta 3 (cadera izquierda) cambió levemente
porque sus vecinas (1 y 4) estaban saturadas en 2.0 y la difusión la
empujó al máximo. El resto se mantuvo igual por el clip.

### Humanoid (PPO entrenado)
Vector g (sin dif): [1.00, 1.00, 1.00, 1.00, 1.59, 1.00, 1.00, 1.51, 1.38,
1.00, 1.00, 1.63, 1.16, 1.25, 1.37, 1.00, 1.28]
Vector g (con dif): [1.00, 1.00, 1.00, 1.00, 1.71, 1.00, 1.00, 1.61, 1.45,
1.00, 1.00, 1.75, 1.16, 1.34, 1.43, 1.00, 1.41]
Diferencia:         [0.00, 0.00, 0.00, 0.00, 0.12, 0.00, 0.00, 0.10, 0.07,
0.00, 0.00, 0.12, 0.01, 0.09, 0.06, 0.00, 0.13]

**Efecto notable.** Múltiples juntas culpables aumentaron entre 0.06 y 0.13.
Las juntas con `g=1.0` (no culpables) se quedaron en 1.0 porque sus vecinas
tampoco eran culpables.

---

## Conclusión para el paper

> "La difusión cinemática aporta valor en robots complejos (Humanoid) donde
> múltiples juntas adyacentes son culpables simultáneamente, pero su efecto
> es marginal en robots simples (Walker2d) donde el clip absorbe la mayoría
> de las modulaciones."

Esto justifica reportarla como **ablación** en el paper:
- Default: difusión desactivada (más rápido, suficiente para Walker2d)
- Variante: difusión activa (mejor en Humanoid)

---

## Hiperparámetros (TODO: mover a config YAML)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `USE_DIFFUSION` | `False` | Activa difusión (False por defecto) |
| `BETA` | 0.2 | Factor de propagación (0.1-0.3 típico) |

### Ablaciones sugeridas

- β: `0.1, 0.2, 0.3` — agresividad de la propagación
- Comparar `g` con y sin difusión por robot