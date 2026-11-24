from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="modelo-riesgo",
    version=Path("modelo_riesgo/VERSION").read_text().strip(),
    description="Modelos de riesgo de siniestros graves por segmento vial (Bogotá).",
    packages=find_packages(),
    package_data={"modelo_riesgo": ["VERSION", "config.yml", "datasets/*", "trained/*"]},
    install_requires=["lightgbm==4.7.0", "numpy>=2.1", "pandas>=2.2.3", "pyyaml>=6"],
    python_requires=">=3.12",
)
