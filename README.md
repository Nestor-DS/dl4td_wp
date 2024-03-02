# **deep_tabular-drinking_water_potability-**
Aplicación de diferentes métodos de deep tabular learning a datasets de problemas del agua usando Python

# Proyecto de Deep Tabular Learning para Calidad del Agua

Este proyecto se centra en la aplicación de métodos de aprendizaje profundo a conjuntos de datos tabulares relacionados con la calidad del agua. Se utilizan diferentes enfoques para la exploración de datos, clasificación con Bosques Aleatorios y técnicas de aprendizaje profundo.

## Estructura del Proyecto

- **dataExploration**: Módulos y notebooks relacionados con la exploración de datos.
  - `exploration/data_exploration.py`: Script para la exploración de datos.
  - `exploration/RandomForestClassifier.py`: Implementación del clasificador de Bosques Aleatorios.

  Notebooks:
  - `I-I(data_exploration).ipynb`: Exploración detallada del conjunto de datos.
  - `II-I(RandomForestClassifier).ipynb`: Implementación y evaluación de Bosques Aleatorios.

- **deep_tabular**: Notebooks relacionados con el aprendizaje profundo tabular.
  - `I-I(hypercubes).ipynb`: Desarrollo de hiperredes y modelos.
  - `I-II(Optimization).ipynb`: Optimización de modelos.
  - `I-III(MetricsEvaluation).ipynb`: Evaluación de métricas de modelos.

- `dptI-I.ipynb`: Notebook principal para la ejecución general del proyecto.
- `drinking_water_potability.csv`: Conjunto de datos sobre la potabilidad del agua.
- `impute_data.py`: Script para la imputación de datos.

## Uso

1. Instala las dependencias utilizando el archivo `requirements.txt`.

    pip install -r requirements.txt

2. Ejecuta el notebook `dptI-I.ipynb` para iniciar el proyecto.

## Dependencias

Asegúrate de tener instaladas las siguientes bibliotecas de Python:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow (para el aprendizaje profundo)

## Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras errores, mejoras potenciales o tienes ideas para nuevas características, no dudes en abrir un problema o enviar una solicitud de extracción.

## Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).
