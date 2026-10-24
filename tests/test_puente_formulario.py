"""Pruebas del puente FORMULARIO entre victimas y siniestros de 2021"""
from pathlib import Path
import pandas as pd
import pytest
from src.exploracion.perfilamiento import puente_formulario

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_cruce_con_datos_simulados():
    # A,B,C,D SIMULAN LOS DATOS Y LAS POSIBILIDADES: 
    # A y B cruzan con CIV valido
    # C cruza pero su siniestro no tiene CIV
    # D solo existe en 2020
    # E no existe, asi que no cruzan.
    siniestros = pd.DataFrame({
        "FORMULARIO": ["A", "B", "C", "D"],
        "ANO_OCURRENCIA_ACC": [2021, 2021, 2021, 2020],
        "CIV": [10, 20, 0, 30],
    })
    victimas = pd.DataFrame({
        "FORMULARIO": ["A", "A", "B", "C", "D", "E"],
        "ANO_OCURRENCIA_ACC": [2021] * 6,
    })
    resultado = puente_formulario(victimas, siniestros)
    assert resultado["victimas_2021"] == 6
    assert resultado["cruzan_con_siniestros"] == 4
    assert resultado["con_civ_valido"] == 3
    assert resultado["formularios_duplicados_en_siniestros"] == 0


def test_detecta_formulario_duplicado():
    siniestros = pd.DataFrame({
        "FORMULARIO": ["A", "A"],
        "ANO_OCURRENCIA_ACC": [2021, 2021],
        "CIV": [10, 10],
    })
    victimas = pd.DataFrame({"FORMULARIO": ["A"], "ANO_OCURRENCIA_ACC": [2021]})
    assert puente_formulario(victimas, siniestros)["formularios_duplicados_en_siniestros"] == 1


@pytest.mark.skipif(
    not (RAW / "victimas_2021_2025.parquet").exists(),
    reason="Faltan los datos de data/raw: correr dvc pull",
)
def test_cruce_con_datos_reales():
    victimas = pd.read_parquet(
        RAW / "victimas_2021_2025.parquet", columns=["FORMULARIO", "ANO_OCURRENCIA_ACC"]
    )
    siniestros = pd.read_parquet(
        RAW / "historico_siniestros.parquet",
        columns=["FORMULARIO", "ANO_OCURRENCIA_ACC", "CIV"],
    )
    resultado = puente_formulario(victimas, siniestros)
    assert resultado["cruzan_pct"] >= 99
    assert resultado["formularios_duplicados_en_siniestros"] == 0