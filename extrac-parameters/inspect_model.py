import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("/home/fernando/rl_env/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets/walker2d.xml")
data = mujoco.MjData(model)

for _ in range(100):
    mujoco.mj_step(model, data)

# Inspeccionar automáticamente todos los atributos de data
for attr in dir(data):
    if attr.startswith("_"):
        continue
    try:
        val = getattr(data, attr)
        if isinstance(val, np.ndarray):
            print(f"  {attr:30s} shape: {val.shape}")
    except:
        pass