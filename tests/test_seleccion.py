"""Pruebas de la seleccion de modelo y del bootstrap pareado."""

import numpy as np
import pandas as pd
import pytest

from src.modelos import seleccion


def test_la_tarea_sale_del_prefijo_del_nombre():
    assert seleccion.tarea_de("lightgbm_poisson_h31_lr0.1_m100") == "conteo"
    assert seleccion.tarea_de("binomial_negativa_eb") == "conteo"
    assert seleccion.tarea_de("linea_base_conteo_historico") == "conteo"
    assert seleccion.tarea_de("lightgbm_binario_h15_lr0.05_m300") == "clasificacion"
    assert seleccion.tarea_de("catboost_d6_lr0.05") == "clasificacion"
    assert seleccion.tarea_de("regresion_logistica") == "clasificacion"


def test_un_nombre_desconocido_no_se_asigna_a_ninguna_tarea():
    """Si manana se anade una familia y se olvida el prefijo, tiene que notarse
    aqui y no colarse dentro de una tarea que no le toca."""
    assert seleccion.tarea_de("red_neuronal_v1") == "desconocida"


def test_la_complejidad_lee_hojas_o_profundidad():
    assert seleccion.complejidad_de("lightgbm_poisson_h31_lr0.1_m100") == 31
    assert seleccion.complejidad_de("lightgbm_binario_h127_lr0.03_m300") == 127
    # Un arbol de profundidad 6 tiene hasta 2**6 hojas: asi las dos familias
    # se comparan en la misma unidad.
    assert seleccion.complejidad_de("catboost_d6_lr0.05") == 64
    # Los GLM no tienen hojas y son los mas simples de todos.
    assert seleccion.complejidad_de("binomial_negativa_glm") == 0


def test_la_captura_de_un_orden_perfecto_es_el_maximo_posible():
    """Diez segmentos, los graves en los dos ultimos. Un orden que los pone
    primeros captura todo lo que cabe en el corte."""
    graves = np.array([0, 0, 0, 0, 0, 0, 0, 0, 3, 5])
    perfecto = graves.astype(float)
    assert seleccion.captura_en(perfecto, graves, k=2) == pytest.approx(100.0)


def test_la_captura_de_un_orden_invertido_es_cero():
    graves = np.array([0, 0, 0, 0, 0, 0, 0, 0, 3, 5])
    invertido = -graves.astype(float)
    assert seleccion.captura_en(invertido, graves, k=2) == pytest.approx(0.0)


def test_la_captura_no_divide_entre_cero_si_no_hubo_graves():
    graves = np.zeros(10, dtype=int)
    assert np.isnan(seleccion.captura_en(np.arange(10.0), graves, k=2))


def _datos(n=600, semilla=3) -> pd.DataFrame:
    """Un modelo bueno, uno igual de bueno y uno malo, sobre los mismos graves."""
    rng = np.random.default_rng(semilla)
    riesgo = rng.gamma(0.4, 1.0, n)
    graves = rng.poisson(riesgo)
    return pd.DataFrame(
        {
            "CIV": np.arange(n),
            "graves_2021": graves,
            "lightgbm_poisson_h31_lr0.1_m100": riesgo + rng.normal(0, 0.05, n),
            "lightgbm_poisson_h15_lr0.1_m100": riesgo + rng.normal(0, 0.05, n),
            "linea_base_conteo_historico": rng.normal(0, 1, n),
        }
    )


def test_el_bootstrap_usa_la_misma_replica_para_todas_las_corridas():
    """Es lo que hace pareada la comparacion. Si cada corrida remuestreara por
    su cuenta, las diferencias saldrian mas anchas de lo que son."""
    datos = _datos()
    corridas = [c for c in datos.columns if c not in seleccion.NO_SON_CORRIDAS]
    a = seleccion.bootstrap(datos, corridas, replicas=40, semilla=7)
    b = seleccion.bootstrap(datos, corridas, replicas=40, semilla=7)
    for c in corridas:
        assert np.allclose(a[c], b[c])  # misma semilla, mismas replicas
    assert len({len(v) for v in a.values()}) == 1


def test_la_diferencia_de_la_lider_consigo_misma_es_cero():
    datos = _datos()
    corridas = [c for c in datos.columns if c not in seleccion.NO_SON_CORRIDAS]
    muestras = seleccion.bootstrap(datos, corridas, replicas=60, semilla=7)
    lider = "lightgbm_poisson_h31_lr0.1_m100"
    comp = seleccion.comparar(muestras, lider)
    ic = comp[lider]["diferencia_con_lider"]
    assert ic["ic95_bajo"] == ic["ic95_alto"] == 0.0
    assert comp[lider]["indistinguible"]


def test_un_modelo_al_azar_si_se_distingue_del_bueno():
    datos = _datos()
    corridas = [c for c in datos.columns if c not in seleccion.NO_SON_CORRIDAS]
    muestras = seleccion.bootstrap(datos, corridas, replicas=200, semilla=7)
    comp = seleccion.comparar(muestras, "lightgbm_poisson_h31_lr0.1_m100")
    assert not comp["linea_base_conteo_historico"]["indistinguible"]


def test_entre_indistinguibles_se_elige_la_mas_simple():
    datos = _datos()
    corridas = [c for c in datos.columns if c not in seleccion.NO_SON_CORRIDAS]
    puntuales = seleccion.metricas_puntuales(datos, corridas)
    muestras = seleccion.bootstrap(datos, corridas, replicas=200, semilla=7)
    lider = "lightgbm_poisson_h31_lr0.1_m100"
    comp = seleccion.comparar(muestras, lider)
    elegida = seleccion.seleccionar(puntuales, muestras, comp, lider)
    # Las dos de LightGBM son estadisticamente iguales por construccion, asi
    # que gana la de 15 hojas sobre la de 31.
    assert elegida["elegida"] == "lightgbm_poisson_h15_lr0.1_m100"


def test_la_linea_base_nunca_se_elige_como_modelo():
    """Aunque empatara, es la referencia contra la que se compara, no un
    candidato: elegirla seria no entregar modelo."""
    datos = _datos()
    corridas = [c for c in datos.columns if c not in seleccion.NO_SON_CORRIDAS]
    puntuales = seleccion.metricas_puntuales(datos, corridas)
    muestras = seleccion.bootstrap(datos, corridas, replicas=60, semilla=7)
    lider = "lightgbm_poisson_h31_lr0.1_m100"
    comp = seleccion.comparar(muestras, lider)
    for nombre in comp:
        comp[nombre]["indistinguible"] = True  # el peor caso: todo empatado
    elegida = seleccion.seleccionar(puntuales, muestras, comp, lider)
    assert elegida["elegida"] != seleccion.LINEA_BASE


def test_las_metricas_puntuales_traen_captura_y_auc_para_toda_corrida():
    """El AUC tambien para las de conteo, que es lo que permite leer la brecha
    entre las dos metricas."""
    datos = _datos()
    corridas = [c for c in datos.columns if c not in seleccion.NO_SON_CORRIDAS]
    puntuales = seleccion.metricas_puntuales(datos, corridas)
    for nombre, fila in puntuales.items():
        assert "auc_roc" in fila
        assert "captura_top_5_pct" in fila
        assert fila["tarea"] in ("conteo", "clasificacion")


def test_la_brecha_cuenta_los_segmentos_que_la_linea_base_no_puede_ordenar():
    """La linea base da el mismo cero a todo segmento sin graves previos. Esa
    es la razon de que su AUC sea bajo aunque su captura sea parecida."""
    datos = pd.DataFrame(
        {
            "CIV": range(10),
            "graves_2021": [0, 0, 0, 0, 1, 0, 0, 2, 0, 1],
            "linea_base_conteo_historico": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 5.0, 1.0, 0.0],
            "lightgbm_poisson_h15_lr0.1_m100": [0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 2.0, 4.0, 1.0, 0.8],
        }
    )
    b = seleccion.brecha_auc_captura(datos, "lightgbm_poisson_h15_lr0.1_m100")
    assert b["segmentos_sin_historia"] == 7  # los siete con cero historico
    assert b["segmentos_sin_historia_pct"] == 70.0
    # Dos de los cuatro graves de 2021 caen donde la linea base no ve nada.
    assert b["graves_2021_sin_historia"] == 2
    assert b["graves_2021_sin_historia_pct"] == 50.0


def test_la_brecha_compara_el_mismo_corte_en_las_dos_metricas():
    datos = _datos()
    datos = datos.rename(columns={"lightgbm_poisson_h15_lr0.1_m100": "modelo"})
    datos["modelo"] = datos["modelo"]
    b = seleccion.brecha_auc_captura(datos, "modelo")
    assert 0.0 <= b["coincidencia_top_5_pct"] <= 100.0
    assert b["auc_modelo"] > b["auc_linea_base"]  # el modelo si ordena, el azar no
    assert b["modelo_comparado"] == "modelo"
