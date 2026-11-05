"""Panel segmento-año (version corta).
 
ENTRA : malla, siniestros, intervenciones (data/raw).
SALE  : data/procesado/panel.parquet, una fila por CIV x ano (2015-2021).
 
PASOS
  1. Malla -> una fila por CIV: suma longitud y exposicion.
  2. Rejilla CIV x año con ceros: sin ella el modelo no ve el 78,6 % sin graves.
  3. Historia: rezago 1 y acumulado PREVIO. Nunca el año que se predice.
  4. Obras: tratado_X (alguna vez) y post_X (años despues de la obra).
 
    python -m src.preparacion.panel
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from src.exploracion.perfilamiento import longitud_proyectada
from src.preparacion.exposicion import exposicion_ampliada, offset_log_longitud
 
RAIZ = Path(__file__).resolve().parents[2]
ANIOS = range(2015, 2022)
 
 
def construir_segmentos(malla: pd.DataFrame) -> pd.DataFrame:
    """Un CIV tiene varios registros: se suman longitud y exposicion."""
    reg = malla[malla.MVICIV != 0].copy()  # CIV 0 = sin llave
    reg["long_m"] = reg.geometry.map(
        lambda g: longitud_proyectada(g) if isinstance(g, str) else np.nan
    )
    clase = pd.to_numeric(reg.MVITCLA, errors="coerce")
    reg["expo_m"] = exposicion_ampliada(reg.long_m, reg.MVINUMC, clase)
 
    seg = reg.groupby("MVICIV").agg(
        long_m=("long_m", "sum"),
        exposicion_m=("expo_m", "sum"),
        carriles=("MVINUMC", "max"),
        clase_via=("MVITCLA", "first"),  # primer valor no vacio
        vel_reg=("MVIVELREG", "first"),
    )
    seg.index = seg.index.astype("int64").rename("CIV")
    seg["clase_via"] = pd.to_numeric(seg.clase_via, errors="coerce")
    seg["vel_reg"] = pd.to_numeric(seg.vel_reg, errors="coerce")
    seg["vel_faltante"] = seg.vel_reg.isna().astype("int8")  # 40 % vacia
    seg["log_long_km"] = offset_log_longitud(seg.long_m)  # offset del modelo
    return seg
 
 
def construir_panel(seg, siniestros, intervenciones) -> pd.DataFrame:
    # 2. Rejilla completa, con 0 donde no hubo graves.
    rejilla = pd.MultiIndex.from_product([seg.index, ANIOS], names=["CIV", "anio"])
    sin = siniestros[siniestros.CIV.isin(seg.index)]
    graves = sin[sin.GRAVEDAD != "SOLO DANOS"]  # grave = no "solo danos"
    cuenta = graves.groupby([graves.CIV.astype("int64"), graves.ANO_OCURRENCIA_ACC]).size()
    cuenta.index.names = ["CIV", "anio"]
    panel = cuenta.reindex(rejilla, fill_value=0).rename("graves").reset_index()
    panel = panel.sort_values(["CIV", "anio"], ignore_index=True)
 
    # 3. Historia: solo anos anteriores (si no, el modelo ve la respuesta).
    por_civ = panel.groupby("CIV").graves
    panel["graves_rezago1"] = por_civ.shift(1)  # 2015 queda vacio
    panel["graves_acum_previo"] = por_civ.cumsum() - panel.graves
 
    # 4. Obras: una columna tratado_ y una post_ por categoria.
    iv = intervenciones[intervenciones.CIV.isin(seg.index)].copy()
    iv["cat"] = (  # "Tráfico  calmado" -> "trafico_calmado"
        iv.CATEGORIA.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    )
    iv["ANIO"] = pd.to_numeric(iv.ANIO, errors="coerce")
    primera = iv.groupby([iv.CIV.astype("int64"), "cat"]).ANIO.min().unstack()
    for cat in primera.columns:
        obra = panel.CIV.map(primera[cat])  # primer ano de obra, o vacio
        panel[f"tratado_{cat}"] = obra.notna().astype("int8")
        panel[f"post_{cat}"] = (panel.anio > obra).astype("int8")
 
    return panel.merge(seg.reset_index(), on="CIV", how="left")
 
 
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrada", type=Path, default=RAIZ / "data" / "raw")
    ap.add_argument("--salida", type=Path, default=RAIZ / "data" / "procesado")
    a = ap.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
 
    malla = pd.read_parquet(a.entrada / "malla_vial_integral.parquet")
    siniestros = pd.read_parquet(a.entrada / "historico_siniestros.parquet")
    obras = pd.read_parquet(a.entrada / "intervenciones_seguridad_vial.parquet")
 
    seg = construir_segmentos(malla)
    panel = construir_panel(seg, siniestros, obras)
    panel.to_parquet(a.salida / "panel.parquet", index=False)
 
    # Control: 116.322 CIV, 814.254 filas, 71.268 graves, 10.717 en 2021.
    print(f"segmentos: {len(seg):,}  |  filas: {len(panel):,}")
    print(f"graves: {int(panel.graves.sum()):,}  |  graves 2021: "
          f"{int(panel.loc[panel.anio == 2021, 'graves'].sum()):,}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())