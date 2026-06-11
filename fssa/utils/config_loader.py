import os
import yaml


def load_config(robot_name):
    """
    Carga el config YAML del robot especificado.

    Parámetros:
        robot_name: "walker2d", "ant" o "humanoid" (lowercase)

    Devuelve:
        dict con la configuración completa
    """
    config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    config_path = os.path.join(config_dir, f"{robot_name}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config no encontrado: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config