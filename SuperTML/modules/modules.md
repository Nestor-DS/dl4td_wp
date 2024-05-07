# Módulos para Proyecto de Aprendizaje Automático (Machine Learning)

Este directorio contiene una colección de módulos Python utilizados en un proyecto de aprendizaje automático llamado "SuperTML". Estos módulos proporcionan funcionalidades para diversas tareas como preparación de datos, entrenamiento de modelos y visualización de resultados.

## Descripción de los Módulos

### `basics.py`

Este módulo proporciona funciones básicas relacionadas con la preparación de datos y la evaluación de modelos. Incluye una función para calcular la puntuación de un conjunto de datos de prueba, así como definiciones de rutas de archivos y configuraciones de visualización.

### `data_cleaning.py`

El módulo `data_cleaning.py` se encarga de la limpieza y preprocesamiento de datos. Contiene funciones para imputar valores faltantes, detectar y eliminar valores atípicos, y normalizar características numéricas en un conjunto de datos.

### `data_preparation.py`

Este archivo contiene funciones para convertir datos tabulares en imágenes utilizando la biblioteca Python Imaging Library (PIL). Las funciones definidas aquí pueden ser útiles para visualizar la estructura y distribución de los datos antes de entrenar modelos de aprendizaje automático.

### `model.py`

En este módulo se define la arquitectura del modelo de aprendizaje automático. Utiliza la biblioteca PyTorch para crear un modelo de red neuronal basado en la arquitectura ResNet-18, adaptado para un problema de clasificación con 3 clases.

### `train.py`

El módulo `train.py` contiene una función para entrenar el modelo definido en `model.py`. Esta función utiliza PyTorch para entrenar el modelo a través de un número especificado de épocas, calculando la pérdida y precisión en conjuntos de datos de entrenamiento y validación.

## Uso

Para utilizar estos módulos, simplemente importa las funciones necesarias en tu script principal de Python y úsalas según sea necesario. Asegúrate de instalar las bibliotecas requeridas, como PyTorch y scikit-learn, antes de ejecutar el código.

## Contribución

Si deseas contribuir a este proyecto, siéntete libre de hacerlo creando una solicitud de extracción (pull request) o abriendo un problema (issue) para discutir posibles mejoras o problemas.


