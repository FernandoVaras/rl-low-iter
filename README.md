# RL Projects - MuJoCo + Gymnasium + Stable-Baselines3

## Requisitos
- Ubuntu 24
- Python 3.12

## 1. Crear entorno virtual
```bash
python3 -m venv ~/rl_env
source ~/rl_env/bin/activate
```

## 2. Instalar dependencias
```bash
pip install --upgrade pip
pip install mujoco gymnasium[mujoco] "stable-baselines3>=2.0" --upgrade
```

## 3. Verificar instalación
```bash
python test_mujoco.py   # verifica que MuJoCo funciona
python test_render.py   # abre ventana visual con acciones aleatorias
```

## 4. Estructura del proyecto

```
rl_projects/
├── walker/
│   ├── train.py
│   ├── demo.py
│   └── results/
├── ant/
│   ├── train.py
│   ├── demo.py
│   └── results/
└── humanoid/
    ├── train.py
    ├── demo.py
    └── results/
```

## 5. Configuración antes de entrenar
En cada `train.py` cambia estas líneas según lo que quieras entrenar:
```python
ALGORITMO = "PPO"    # ← PPO, SAC
USE_GSDE  = False    # ← True para activar gSDE
TOTAL_STEPS = 1_000_000
```

## 6. Entrenar
Siempre activa el entorno virtual primero:
```bash
source ~/rl_env/bin/activate
```

Luego entra a la carpeta del robot y entrena:
```bash
cd ~/rl_projects/walker
python train.py

cd ~/rl_projects/ant
python train.py

cd ~/rl_projects/humanoid
python train.py
```

## 7. Parar y continuar el entrenamiento
Puedes interrumpir el entrenamiento cuando quieras con:
Ctrl+C
El modelo se guarda automáticamente en `results/` con los pasos acumulados.
La próxima vez que corras `train.py` continuará desde donde quedó hasta llegar a `TOTAL_STEPS`.

También puede ocurrir que el entrenamiento se quede colgado o no termine,
en ese caso usa `Ctrl+C` para forzar el guardado y salir limpiamente.

## 8. Ver demo del modelo entrenado
En `demo.py` configura el mismo ALGORITMO y USE_GSDE que usaste al entrenar:
```python
ALGORITMO = "PPO"
USE_GSDE  = False
DEMO_STEPS = 3000   # ← cuántos pasos dura la demo visual
```

Luego corre en otra terminal (con el entorno activado):
```bash
cd ~/rl_projects/walker
python demo.py
```

Puedes ver la demo mientras el entrenamiento sigue corriendo en otra terminal.

## 9. Resultados
Cada entrenamiento genera en `results/`:
- `Walker2d_PPO.zip`          → modelo guardado
- `Walker2d_PPO_pasos.txt`    → pasos acumulados
- `grafica_Walker2d_PPO.png`  → curva de aprendizaje

El nombre cambia según ROBOT, ALGORITMO y USE_GSDE:
- `Walker2d_PPO.zip`
- `Walker2d_PPO_gSDE.zip`
- `Walker2d_SAC.zip`
- `Ant_PPO.zip`
- `Humanoid_SAC_gSDE.zip`
- ...