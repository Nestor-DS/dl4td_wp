import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

from lumin.plotting.plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("whitegrid")

DATA_PATH     = Path("../data/")
IMG_PATH      = Path("../images_saved/")
RESULTS_PATH  = Path("../results/")
plot_settings = PlotSettings(cat_palette='tab10', savepath=Path('.'), format='.pdf')



def hi():
    print ("hi")
    
def score_test_df(df:pd.DataFrame, cut:float, public_wgt_factor:float=1, private_wgt_factor:float=1, verbose:bool=True):
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
