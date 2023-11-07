import yaml
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import *
from models.logistic_regression import MultiClassLogisticRegression 

if __name__ == '__main__':

    with open('./config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    
    model_name = config['training']['model_name']
    
    df_train = pd.read_csv(config['dataset']['train_csv'])
    df_train = df_train[['lat', 'lon', 'TMQ', 'U850', 'V850', 'UBOT', 'VBOT', 'QREFHT',
       'PS', 'PSL', 'T200', 'T500', 'PRECT', 'TS', 'TREFHT', 'Z1000', 'Z200',
       'ZBOT', 'time', 'Label']]

    df_train = add_feats(df_train, config['dataset'])

    feats = [x for x in df_train.columns if x not in ['Label', 'lat', 'lon','time']]
    
    if model_name=='logistic_regression':
        X,y = get_data(df_train, feats, model_name=model_name, is_test=False)
        X = normalise(X)

        hyperparams = config['training']['logistic_regression']
        clf = MultiClassLogisticRegression(
            is_weighted=hyperparams['is_weighted_loss'], 
            learning_rate=hyperparams['learning_rate'], 
            lambda1=hyperparams['lambda1'], 
            lambda2=hyperparams['lambda2'], 
            num_epochs=hyperparams['num_epochs']
            )
        best_W, best_b = clf.fit(X.T, y, X.T, y)