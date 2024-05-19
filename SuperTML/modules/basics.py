# Librerías ocupadas

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn para la manipulación de datos y métricas
from sklearn.model_selection import train_test_split   # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.datasets import load_iris                 # Para cargar el conjunto de datos de Iris
from sklearn.metrics import roc_auc_score              # Para calcular el ROC AUC

# Importar módulos de PyTorch para la creación y entrenamiento de redes neuronales
import torch                       # Biblioteca PyTorch
import torch.nn as nn              # Módulo de redes neuronales
import torch.nn.functional as F    # Funciones de activación y otras operaciones para redes neuronales
import torch.optim as optim        # Algoritmos de optimización
from torch.utils.data import DataLoader, TensorDataset  # Para crear datasets y data loaders personalizados

# Importar módulos de imbalanced-learn para manejar el desbalance de clases
from imblearn.over_sampling import RandomOverSampler

# Importar módulos específicos de Lumin para la configuración de gráficos
from lumin.plotting.plot_settings import PlotSettings

# Importar módulos adicionales
from pathlib import Path  # Para manipulación de rutas de archivos


# Configuración de estilo de seaborn para la visualización
sns.set_style("whitegrid")

# Rutas para los datos, imágenes y resultados
DATA_PATH     = Path("../data/")
IMG_PATH      = Path("../images_saved/")
RESULTS_PATH  = Path("../results/")

# Configuración para las gráficas
plot_settings = PlotSettings(cat_palette='tab10', savepath=Path('.'), format='.pdf')

# Función para calcular el puntaje de prueba
def score_test_df(df:pd.DataFrame, cut:float, public_wgt_factor:float=1, private_wgt_factor:float=1, verbose:bool=True):
    """
    Calcula el puntaje de prueba de un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame con las predicciones y otros datos relevantes.
        cut (float): Umbral de corte para las predicciones.
        public_wgt_factor (float): Factor de peso para el cálculo del AMS público.
        private_wgt_factor (float): Factor de peso para el cálculo del AMS privado.
        verbose (bool): Indica si se mostrará información detallada.
    
    Returns:
        public_ams (float): Puntaje AMS público.
        private_ams (float): Puntaje AMS privado.
    """
    accept = (df.pred >= cut)
    signal = (df.gen_target == 1)
    bkg = (df.gen_target == 0)
    public = (df.private == 0)
    private = (df.private == 1)

    public_ams = calc_ams(public_wgt_factor*np.sum(df.loc[accept & public & signal, 'gen_weight']),
                          public_wgt_factor*np.sum(df.loc[accept & public & bkg, 'gen_weight']))

    private_ams = calc_ams(private_wgt_factor*np.sum(df.loc[accept & private & signal, 'gen_weight']),
                           private_wgt_factor*np.sum(df.loc[accept & private & bkg, 'gen_weight']))

    if verbose: print("Public:Private AMS: {} : {}".format(public_ams, private_ams))    
    return public_ams, private_ams
