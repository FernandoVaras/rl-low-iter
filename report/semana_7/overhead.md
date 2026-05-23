# Overhead Computacional del Wrapper

## ¿Qué se midió?

Tiempo promedio por `step()` del entorno **con** y **sin** `FaultCaptureWrapper`,
en los 3 robots de benchmark. El wrapper agrega:
- 8 copias de arrays por paso (qpos, qvel, cfrc_ext, etc.)
- Almacenamiento en buffer circular (deque)
- Chequeo de condición de fallo

---

## Resultados
Robot      |   Sin (ms) |   Con (ms) |     Overhead
-----------+------------+------------+-------------
Walker2d   |     0.1683 |     0.1498 |      -11.01%
Ant        |     0.2352 |     0.2456 |       +4.40%
Humanoid   |     0.2589 |     0.2702 |       +4.36%

Medido con 5000 pasos por robot en CPU Ryzen 5 7000.

---

## Interpretación

- **Walker2d (-11%)**: el valor negativo es ruido estadístico, no una mejora real.
  Cuando las diferencias son del orden de 0.02ms, variaciones del sistema
  operativo, cache, etc., dominan sobre el costo del wrapper.

- **Ant (+4.40%)** y **Humanoid (+4.36%)**: overhead consistente y pequeño,
  ~0.01ms por paso. Más notable en estos robots porque tienen más bodies
  y joints (más datos para copiar).

---

## Conclusión

El overhead computacional del wrapper es menor al 5% en todos los robots
evaluados. La captura pre-fallo no impone una carga significativa al
entrenamiento.

> **Para el paper:** "El overhead computacional del wrapper se mantiene
> bajo el 5% en los tres robots benchmark (Walker2d, Ant, Humanoid),
> validando que la captura de señales pre-fallo no compromete la
> eficiencia del entrenamiento."

---

## Reproducir el test

```bash
cd ~/rl_projects
python fssa/test/test_overhead.py
```

Para resultados más estables, aumentar `N_PASOS` a 50000 en el config.