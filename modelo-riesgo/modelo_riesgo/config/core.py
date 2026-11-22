from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PACKAGE_ROOT / "datasets"
TRAINED = PACKAGE_ROOT / "trained"

# Variables, hiperparámetros y nombres de archivo del modelo (lo escribe src/modelos/empaquetar.py).
config = yaml.safe_load((PACKAGE_ROOT / "config.yml").read_text(encoding="utf-8"))