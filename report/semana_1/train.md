# Guía de Entrenamiento

## Algoritmos

### PPO (Proximal Policy Optimization)
Algoritmo on-policy, aprende solo de experiencias recientes y las descarta.
Agrega ruido aleatorio en cada paso para explorar, lo que a veces genera
movimientos espasmódicos pero aprende de forma estable.

### SAC (Soft Actor-Critic)
Algoritmo off-policy, guarda todas las experiencias en un replay buffer
y aprende de experiencias pasadas y presentes mezcladas. Busca dos objetivos
simultáneamente: maximizar recompensa y maximizar entropía (seguir explorando).
Más eficiente que PPO, llega a buena recompensa con menos pasos pero
es más lento por paso porque actualiza 3 redes simultáneamente.

### gSDE (Generalized State-Dependent Exploration)
Modificación de la exploración que se puede acoplar tanto a PPO como a SAC.
En vez de agregar ruido aleatorio en cada paso, el ruido depende del estado
actual del robot y se mantiene consistente por varios pasos seguidos.

**PPO + gSDE** → en vez de temblar cada paso, elige una estrategia y la
ejecuta completa hasta fallar. Al inicio genera episodios más cortos pero
el aprendizaje es más limpio y los movimientos más naturales.

**SAC + gSDE** → SAC ya explora bien por su entropía máxima, agregar gSDE
le da exploración adicional coherente con el estado. Puede mejorar
la suavidad de movimiento en entornos de locomoción.

---

## Configuración

En cada `train.py` cambia estas líneas:

```python
ALGORITMO = "PPO"       # ← PPO, SAC
USE_GSDE = False        # ← True para activar gSDE
TOTAL_STEPS = 700_000   # ← igual para todos los modelos
```

---

## Correr entrenamiento

Siempre activa el entorno virtual primero:
```bash
source ~/rl_env/bin/activate
```

Luego entra a la carpeta del robot:
```bash
cd ~/rl_projects/walker
python train.py

cd ~/rl_projects/ant
python train.py

cd ~/rl_projects/humanoid
python train.py
```

---

## Interrumpir y continuar

Puedes parar cuando quieras con `Ctrl+C`, el modelo se guarda automáticamente
con los pasos acumulados. Al volver a correr `train.py` continúa desde donde quedó.

1ra corrida: para en 200k (Ctrl+C) → guarda
2da corrida: continúa de 200k → para en 500k (Ctrl+C) → guarda
3ra corrida: continúa de 500k → llega a 700k → termina solo

Si el entrenamiento se queda colgado o no termina, usa `Ctrl+C` para
forzar el guardado y salir limpiamente.

---

## Ver demo mientras entrena

Abre una segunda terminal, activa el entorno y corre:
```bash
source ~/rl_env/bin/activate
cd ~/rl_projects/walker
python demo.py
```

En `demo.py` configura el mismo `ALGORITMO` y `USE_GSDE` que usaste al entrenar.

---

## Resultados generados

Cada entrenamiento guarda en `results/`:

| Archivo | Descripción |
|---------|-------------|
| `Walker2d_PPO.zip` | modelo guardado |
| `Walker2d_PPO_pasos.txt` | pasos acumulados para continuar |
| `grafica_Walker2d_PPO.png` | curva de aprendizaje |
| `Walker2d_PPO_entrenamiento.csv` | datos por episodio (episodio, pasos, recompensa) |
