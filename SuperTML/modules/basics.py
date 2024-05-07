import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

# Importaciones específicas de Lumin
from lumin.plotting.plot_settings import PlotSettings

# Importaciones para la visualización
import seaborn as sns
import matplotlib.pyplot as plt

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
