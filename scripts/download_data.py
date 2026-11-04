"""Descarga (o genera) los datasets del curso y los guarda como CSV en data/.

Uso, desde la raíz del repo:

    uv run python scripts/download_data.py

Puedes ejecutarlo todas las veces que quieras: vuelve a crear los archivos.
La carpeta data/ está en el .gitignore, así que los datos no se suben a GitHub.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import truststore
from sklearn import datasets

# Usa los certificados del sistema operativo para las descargas HTTPS. Sin esto,
# el Python de python.org en macOS falla con "CERTIFICATE_VERIFY_FAILED".
truststore.inject_into_ssl()

# Carpeta data/ en la raíz del repo (este archivo está en scripts/)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generar_viviendas():
    """Ventas de viviendas sintéticas, sin errores (capítulo 3: pandas)."""
    rng = np.random.default_rng(42)
    n = 600

    barrio = rng.choice(["Centro", "Norte", "Sur", "Este", "Oeste"], size=n, p=[0.2, 0.25, 0.2, 0.15, 0.2])
    metros = rng.normal(90, 30, n).clip(30, 250).round()
    habitaciones = np.clip(metros // 30 + rng.integers(-1, 2, n), 1, 6).astype(int)
    antiguedad = rng.integers(0, 60, n).astype(float)
    garaje = rng.choice(["sí", "no"], size=n, p=[0.4, 0.6])
    fecha = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")
    precio_m2 = pd.Series(barrio).map({"Centro": 3500, "Norte": 2800, "Sur": 2000, "Este": 2400, "Oeste": 2600}).to_numpy()
    precio = (metros * precio_m2 + habitaciones * 5000 - antiguedad * 400
              + (garaje == "sí") * 12000 + rng.normal(0, 15000, n)).round(-2)

    datos = pd.DataFrame({
        "id": np.arange(1, n + 1), "fecha_venta": fecha.strftime("%Y-%m-%d"), "barrio": barrio,
        "metros": metros, "habitaciones": habitaciones, "antiguedad": antiguedad,
        "garaje": garaje, "precio": precio,
    })
    return datos


def generar_viviendas_sucio():
    """Las mismas viviendas con errores a propósito, para practicar la limpieza de datos.

    Tiene valores faltantes, filas duplicadas, texto inconsistente, precios
    guardados como texto y errores de tipeo en el precio, como los datos reales.
    Además añade dos columnas para practicar transformaciones: una muy
    asimétrica (distancia_metro_m) y una categórica ordinal (estado).
    """
    datos = generar_viviendas()
    n = len(datos)

    # Columnas extra
    rng_extra = np.random.default_rng(11)
    mediana_distancia = datos["barrio"].map({"Centro": 300, "Norte": 900, "Sur": 1500, "Este": 1100, "Oeste": 1200})
    datos["distancia_metro_m"] = (mediana_distancia * rng_extra.lognormal(0, 0.8, n)).round()
    deterioro = datos["antiguedad"] + rng_extra.normal(0, 12, n)
    datos["estado"] = pd.cut(deterioro, bins=[-np.inf, 15, 40, np.inf], labels=["como nuevo", "buen estado", "a reformar"]).astype(str)
    datos = datos[["id", "fecha_venta", "barrio", "metros", "habitaciones", "antiguedad",
                   "garaje", "estado", "distancia_metro_m", "precio"]]

    # Ensuciamos los datos como en la vida real
    rng = np.random.default_rng(7)
    datos.loc[rng.choice(n, 30, replace=False), "metros"] = np.nan
    datos.loc[rng.choice(n, 20, replace=False), "antiguedad"] = np.nan
    datos.loc[rng.choice(n, 8, replace=False), "barrio"] = np.nan
    idx = rng.choice(n, 25, replace=False)
    datos.loc[idx[:10], "barrio"] = datos.loc[idx[:10], "barrio"].str.upper()
    datos.loc[idx[10:], "barrio"] = "  " + datos.loc[idx[10:], "barrio"].str.lower() + " "
    datos.loc[rng.choice(n, 4, replace=False), "precio"] *= 10       # errores de tipeo
    datos.loc[rng_extra.choice(n, 15, replace=False), "estado"] = np.nan
    datos["precio"] = datos["precio"].astype(object)                  # algunos precios como texto: "$245,000"
    idx_texto = rng_extra.choice(n, 20, replace=False)
    datos.loc[idx_texto, "precio"] = [f"${p:,.0f}" for p in datos.loc[idx_texto, "precio"]]
    datos = pd.concat([datos, datos.sample(12, random_state=1)])      # filas duplicadas
    datos = datos.sample(frac=1, random_state=2)                      # desordenar
    return datos


# Nombre del archivo → función que devuelve el DataFrame
DATASETS = {
    # Sintético
    "viviendas.csv": generar_viviendas,
    "viviendas_sucio.csv": generar_viviendas_sucio,
    # seaborn (se descargan de internet)
    "titanic.csv": lambda: sns.load_dataset("titanic"),
    "penguins.csv": lambda: sns.load_dataset("penguins"),
    "tips.csv": lambda: sns.load_dataset("tips"),
    "mpg.csv": lambda: sns.load_dataset("mpg"),
    # scikit-learn (iris y breast_cancer vienen incluidos; california_housing se descarga)
    "iris.csv": lambda: datasets.load_iris(as_frame=True).frame,
    "breast_cancer.csv": lambda: datasets.load_breast_cancer(as_frame=True).frame,
    "california_housing.csv": lambda: datasets.fetch_california_housing(as_frame=True).frame,
}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    fallidos = []

    for nombre, cargar in DATASETS.items():
        try:
            df = cargar()
        except Exception as error:  # por ejemplo, sin conexión a internet
            print(f"✘ data/{nombre}: {error}")
            fallidos.append(nombre)
            continue
        df.to_csv(DATA_DIR / nombre, index=False)
        print(f"✔ data/{nombre:<24} {len(df):>6} filas × {df.shape[1]} columnas")

    if fallidos:
        raise SystemExit(f"\nNo se pudieron descargar: {', '.join(fallidos)}. "
                         "Revisa tu conexión a internet y vuelve a ejecutar el script.")


if __name__ == "__main__":
    main()
