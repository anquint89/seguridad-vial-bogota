"""La metrica de negocio, con los empates resueltos de forma explicita.

QUE MIDE
  Que fraccion de los siniestros graves del anio de validacion cae en los k
  primeros segmentos del orden que propone un modelo. Es la misma para las dos
  tareas y para la linea base, que es lo que las hace comparables.

POR QUE NO BASTA ORDENAR Y CORTAR
  El metodo vigente ordena por el conteo historico, que es un entero pequeno.
  En el corte del 5 % el segmento numero 5.816 tiene tres graves previos y hay
  1.881 segmentos empatados en ese mismo valor. Cual de ellos entra al top
  depende del orden en que el algoritmo de ordenamiento los deje, y ese orden
  no esta definido.

  Medido sobre los datos de 2021, la captura de la linea base va de 41,21 a
  45,03 segun como se desempate. Con 200 desempates aleatorios sale 44,44 de
  media. Es decir, casi cuatro puntos de diferencia por una decision que nadie
  tomo, sobre una cifra que es el punto de comparacion de todo el trabajo.

  Los modelos no sufren esto porque sus puntajes son continuos y practicamente
  no empatan. El problema es solo de la linea base, que es justamente contra lo
  que se mide la mejora.

QUE SE HACE EN CAMBIO
  Los segmentos que superan estrictamente el valor de corte entran enteros. Del
  grupo empatado en el corte se toma la parte proporcional que cabe. Eso es
  exactamente el valor esperado bajo desempate aleatorio, es determinista y no
  depende de ningun orden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def captura_top(puntaje, graves, k: int) -> float:
    """Porcentaje de los graves que cae en los k mejores segmentos del orden.

    `puntaje` y `graves` se alinean por posicion. Del grupo empatado en el
    valor de corte se toma la parte proporcional, que es el valor esperado bajo
    desempate aleatorio.
    """
    p = np.asarray(puntaje.to_numpy() if isinstance(puntaje, pd.Series) else puntaje, dtype=float)
    g = np.asarray(graves.to_numpy() if isinstance(graves, pd.Series) else graves, dtype=float)
    total = g.sum()
    if total <= 0 or k <= 0:
        return float("nan")
    if k >= len(p):
        return 100.0

    # argpartition deja en la posicion k-1 el valor que ocupa ese puesto, en
    # tiempo lineal. Con quinientas replicas por sesenta corridas, ordenar
    # entero cuesta minutos.
    corte = p[np.argpartition(-p, k - 1)[k - 1]]
    estrictos = p > corte
    empatados = p == corte
    n_empatados = int(empatados.sum())
    cupo = k - int(estrictos.sum())

    capturado = g[estrictos].sum()
    if n_empatados > 0 and cupo > 0:
        capturado += (cupo / n_empatados) * g[empatados].sum()
    return 100.0 * capturado / total


def captura_top_alineada(puntaje: pd.Series, graves: pd.Series, k: int) -> float:
    """Igual que captura_top, pero alinea por el indice en vez de por posicion.

    La linea base se agrupa por CIV y sale en otro orden que las predicciones.
    Alinear por posicion ahi compararia segmentos distintos sin avisar. Un CIV
    que no este en el puntaje queda en cero, que es lo que corresponde: el
    metodo vigente no sabe ordenarlo.
    """
    alineado = puntaje.reindex(graves.index).fillna(0.0)
    return captura_top(alineado, graves, k)


def empates_en_el_corte(puntaje, k: int) -> dict:
    """Cuantos segmentos empatan en el valor de corte y cuantos caben.

    Sirve para declarar en el reporte si una cifra depende del desempate.
    """
    p = np.asarray(puntaje.to_numpy() if isinstance(puntaje, pd.Series) else puntaje, dtype=float)
    if k <= 0 or k >= len(p):
        return {"valor_de_corte": None, "empatados": 0, "cupo": 0}
    corte = p[np.argpartition(-p, k - 1)[k - 1]]
    estrictos = int((p > corte).sum())
    return {
        "valor_de_corte": float(corte),
        "empatados": int((p == corte).sum()),
        "cupo": k - estrictos,
    }
