# Plan de definición de la ventana pre-fallo

## Contexto

El módulo de ponderación del ruido de exploración necesita capturar una ventana de pasos previos al evento de fallo (`terminated=True`) para calcular el vector culpa por junta. La forma de definir esta ventana afecta directamente la calidad de la firma sensorial extraída.

Existen tres regiones temporales conceptuales en cualquier episodio que termina en fallo:

```
[───── lectura normal ─────][── lectura rara ──][── ya cayendo ──] FALLO (terminated=True)
```

- **Lectura normal**: el robot opera dentro de su comportamiento aprendido.
- **Lectura rara**: aparecen las primeras desviaciones — aquí está la *causa* del fallo.
- **Ya cayendo**: el sistema está fuera de control — esto es *consecuencia*, no causa.

## Decisión

Se adopta un enfoque incremental:

**Fase 1 (inicial):** ventana fija sin offset. Se toman los últimos N pasos antes del fallo, incluyendo la zona de "ya cayendo".

**Fase 2 (si los resultados lo justifican):** ventana fija con offset terminal. Se excluyen los últimos K pasos para capturar solo causa.

## Fase 1 — Sin offset

### Definición

Ventana = `[t_fallo - N, t_fallo]` con N = 50 pasos como valor inicial.

### Por qué se espera que funcione

1. **Dilución por integración temporal.** El Jacobiano se evalúa en múltiples instantes de la ventana y se agrega. Los pasos de "ya cayendo" son una fracción minoritaria del total, por lo que su señal entra pero no domina.

2. **Robustez por múltiples Jacobianos.** El módulo usa Jacobianos de pie izquierdo, pie derecho y centro de masa. Una junta verdaderamente causal aparece como culpable en varios; una junta que solo aparece culpable durante la caída final destaca dominantemente en uno solo. La combinación atenúa el ruido de la fase de consecuencia.

3. **Reasignación por propagación cinemática.** El operador de difusión sobre el grafo cinemático redistribuye la culpa entre juntas vecinas. Si la cadera aparece sobrerepresentada en la fase final pero el verdadero responsable era el pie, parte de la culpa migra hacia el pie a través de la cadena.

### Por qué podría no funcionar

1. **Sesgo hacia juntas proximales al torso.** En la fase de "ya cayendo", la cadera y el torso siempre presentan inclinación marcada simplemente por la geometría de la caída. Sin offset, esto puede generar un vector g donde la cadera siempre tiene peso alto, independiente de cuál fue la causa real. El módulo degeneraría a "siempre explorar más en la cadera", perdiendo poder discriminativo.

2. **Saturación numérica.** Los últimos pasos antes del fallo presentan valores extremos en velocidad articular, aceleración y fuerzas de constraint. Estos valores pueden dominar numéricamente la pseudoinversa del Jacobiano y sesgar el cálculo del vector culpa.

3. **Sensibilidad al tipo de fallo.** En fallos lentos (pérdida gradual de estabilidad), la fase rara es larga y diluye bien la consecuencia. En fallos bruscos (resbalón súbito), la fase rara es corta y la consecuencia domina. Sin offset, los fallos bruscos quedan analizados con peor calidad.

### Criterios para detectar problemas

Durante el entrenamiento se monitorea:

- Histograma del vector g promedio por junta. Si está siempre dominado por las mismas juntas proximales, hay sesgo de consecuencia.
- Distribución de g entre tipos de fallo. Si todos los fallos producen vectores g similares, el módulo no está discriminando información sensorial — solo está leyendo "el robot cayó".
- Estabilidad numérica del cálculo. Si aparecen valores fuera de rango en el vector culpa antes de aplicar cotas, hay saturación.

## Fase 2 — Con offset (contingente)

### Activación

Se pasa a Fase 2 si durante Fase 1 se observa cualquiera de los síntomas anteriores, o si el desempeño del módulo no supera consistentemente al baseline.

### Definición

Ventana = `[t_fallo - N - K, t_fallo - K]` con K = 10 pasos como offset inicial.

Esto excluye los últimos K pasos (zona de "ya cayendo") y captura los N pasos previos a esa zona.

### Por qué podría funcionar mejor

1. La firma sensorial captura solo la fase de causa, sin contaminación por la consecuencia geométrica del colapso.
2. Reduce el sesgo hacia juntas proximales al torso.
3. Mitiga la saturación numérica al evitar los valores extremos del desenlace.

### Por qué podría no aportar mejora

1. **Pérdida de información útil.** Algunos fallos no son separables limpiamente entre causa y consecuencia. Excluir los últimos K pasos puede eliminar información que sí era diagnóstica.
2. **Calibración de K es no trivial.** K demasiado pequeño no resuelve el problema; K demasiado grande pierde la causa real. Esto añade un hiperparámetro a tunear.
3. **Fallos cortos quedan peor manejados.** Si un episodio dura solo 30 pasos y K=10, la ventana queda con 20 pasos efectivos, reduciendo la calidad estadística de la firma.

## Resumen del plan

| Fase | Ventana | Cuándo se usa | Riesgo |
|------|---------|---------------|--------|
| 1 | `[t-50, t]` | Por defecto, primera implementación | Sesgo hacia juntas proximales |
| 2 | `[t-60, t-10]` | Si Fase 1 muestra sesgo o bajo desempeño | Pérdida de información, K a calibrar |

## Implicación para los experimentos

En la sección de ablaciones del paper se reportará la comparación entre ambas configuraciones, independientemente de cuál se use como diseño principal. Esto permite:

- Validar empíricamente la elección.
- Mostrar al revisor que el método es robusto a la configuración de ventana.
- Documentar bajo qué condiciones cada variante es preferible.

## Notas adicionales

- La señal de fallo es exclusivamente `terminated=True` de Gymnasium; no se modifica el entorno base.
- La ventana solo se procesa si el episodio duró al menos N pasos (warm-up). Episodios más cortos no activan el módulo.
- Los valores N=50 y K=10 son iniciales; pueden ajustarse durante el tuning de hiperparámetros.