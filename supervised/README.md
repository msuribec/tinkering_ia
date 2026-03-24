# Taller 02 - Aprendizaje Supervisado

Repositorio del taller de aprendizaje supervisado con dos pipelines completos:

- Regresion sobre `Medical Cost Insurance`
- Clasificacion multiclase sobre `Digits`

La entrega incluye notebooks, aplicacion en Streamlit y datasets versionados localmente para que el proyecto sea reproducible sin depender de URLs externas.

## Estructura del proyecto

```text
Workshop_02/
├── app/
│   ├── app.py
│   ├── pages/
│   └── utils/
├── data/
│   └── raw/
│       ├── insurance.csv
│       └── digits.csv
├── notebooks/
│   ├── regresion.ipynb
│   └── clasificacion.ipynb
├── requirements.txt
└── README.md
```

## Setup local

El proyecto fue validado con el entorno virtual local `venv`.

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar la app

Desde la raiz del repositorio:

```bash
streamlit run app/app.py
```

## Evidencia solicitada en la Parte 3

La aplicacion incluye:

- Prediccion individual
- Prediccion por lote mediante carga de CSV
- Descarga de CSV de ejemplo y CSV con resultados
- Visualizacion de feature importance para modelos basados en arboles
- Evidencia de 5 casos documentados de prueba

## Deployment en Streamlit Cloud

Configuracion recomendada:

- Repository root: este repositorio
- Main file path: `app/app.py`
- Requirements file: `requirements.txt`
- Python version: 3.9

## URL de la aplicacion desplegada

Pendiente de agregar despues del despliegue final en Streamlit Cloud:

`STREAMLIT_CLOUD_URL_AQUI`
