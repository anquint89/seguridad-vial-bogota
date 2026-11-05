"""Proxy de exposicion (las capas publicas no traen aforos vehiculares)
 
  log_long_km  : log de la longitud en km. Es el offset del modelo: un tramo
                 largo espera mas siniestros solo por ser largo.
  exposicion_m : longitud x carriles x peso de la clase de via. Sirve para
                 comparar contra la base como sensibilidad en MLflow.
 
Los pesos son un SUPUESTO: no se estiman con los siniestros (seria circular).
VALIDAR el significado de los codigos de MVITCLA con el diccionario de datos.
"""
 
import numpy as np
import pandas as pd
 
PESO_CLASE = {1: 3.0, 2: 2.0, 3: 1.0}  # supuesto: 1 = via mas importante
PESO_DEFECTO = 1.0                     # clases 4-7 y vacios
LONGITUD_MINIMA_KM = 0.001             # 1 m: el log de 0 no existe
 
 
def offset_log_longitud(long_m: pd.Series) -> pd.Series:
    """log(km) con piso de 1 m. 1.000 m -> 0 | 500 m -> -0,69 | 0 m -> -6,91."""
    return np.log(np.maximum(long_m / 1000.0, LONGITUD_MINIMA_KM))
 
 
def exposicion_ampliada(long_m: pd.Series, carriles: pd.Series, clase: pd.Series) -> pd.Series:
    """Longitud x carriles x peso. 1.000 m, 2 carriles, clase 1 -> 1.000 x 2 x 3 = 6.000."""
    peso = pd.to_numeric(clase, errors="coerce").map(PESO_CLASE).fillna(PESO_DEFECTO)
    return long_m * carriles * peso