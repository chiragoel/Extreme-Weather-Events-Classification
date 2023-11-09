import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn import svm

from utils import *
from models.models_main import *
from models.logistic_regression import MultiClassLogisticRegression

def train_logistic_regression(df_train, feats, training_hyperparams):
    '''
        Train the Logistic regression model
        Args:
            df_train: The pandas dataframe
            feats: List of all the features to use for model training
            training_hyperparams: Hyperparameters for the model
        Returns trained weights
    '''
    X,y = get_data(df_train, feats, model_name=model_name, is_test=False)
    X = normalise(X)

    hyperparams = training_hyperparams
    if training_hyperparams['is_cv']:
        X_train, X_val = split_dataset(X)
        y_train, y_val = split_dataset(y)
        hyperparams['learning_rate'], hyperparams['lambda1'], hyperparams['lambda2'], hyperparams['num_epochs']  = cv_logistic_regression(X_train, y_train, X_val, y_val)

    if training_hyperparams['is_training_curve']:
        clf = MultiClassLogisticRegression(
            is_weighted=hyperparams['is_weighted_loss'], 
            learning_rate=hyperparams['learning_rate'], 
            lambda1=hyperparams['lambda1'], 
            lambda2=hyperparams['lambda2'], 
            num_epochs=hyperparams['num_epochs']
            )
        X_train, X_val = split_dataset(X)
        y_train, y_val = split_dataset(y)
        _,_ = clf.fit(X_train.T, y_train, X_val.T, y_val,plot_curves=True)
    
    clf = MultiClassLogisticRegression(
        is_weighted=hyperparams['is_weighted_loss'], 
        learning_rate=hyperparams['learning_rate'], 
        lambda1=hyperparams['lambda1'], 
        lambda2=hyperparams['lambda2'], 
        num_epochs=hyperparams['num_epochs']
        )

    best_W, best_b = clf.fit(X.T, y, X.T, y)
    return best_W, best_b

def test_logistic_regression(df_test, feats, weights, bias):
    '''
        Testing function for logistic regression
        Args:
            df_test: The test dataframe
            feats: The list of features to use
            weights: Trained weights
            bias: Trained bias
        Returns predicted output
    '''
    X = get_data(df_test, feats, model_name=model_name, is_test=True)
    X = normalise(X)
    Z = weights.T.dot(X.T) + bias
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extreme Weather Classification')
    parser.add_argument('--config_path', type=str, default='./config.yaml')
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    
    model_name = config['training']['model_name']
    
    #training setup
    df_train = pd.read_csv(config['dataset']['train_csv'])
    df_train = df_train[['lat', 'lon', 'TMQ', 'U850', 'V850', 'UBOT', 'VBOT', 'QREFHT',
       'PS', 'PSL', 'T200', 'T500', 'PRECT', 'TS', 'TREFHT', 'Z1000', 'Z200',
       'ZBOT', 'time', 'Label']]

    df_train = add_feats(df_train, config['dataset'], model_name)

    #testing setup
    df_test = pd.read_csv(config['dataset']['test_csv'])
    df_test_ = df_test[['lat', 'lon', 'TMQ', 'U850', 'V850', 'UBOT', 'VBOT', 'QREFHT',
       'PS', 'PSL', 'T200', 'T500', 'PRECT', 'TS', 'TREFHT', 'Z1000', 'Z200',
       'ZBOT', 'time']]

    df_test_ = add_feats(df_test_, config['dataset'], model_name)

    if model_name=='logistic_regression':
        
        print('Training Logistic Regression')
        feats = [x for x in df_train.columns if x not in ['Label', 'lat', 'lon','time']]
        W,b = train_logistic_regression(df_train, feats, config['training']['logistic_regression'])
        y_pred = np.argmax(test_logistic_regression(df_test_, feats, W, b), axis=0)
        df_test['Label'] = y_pred
        df_submit = df_test[['SNo', 'Label']]
        df_submit.to_csv(os.path.join(config['test']['save_dir'], 'result.csv'), index=False)
        np.save(os.path.join(config['test']['save_dir'], 'W.npy'), W)
        np.save(os.path.join(config['test']['save_dir'], 'b.npy'), b)
    
    elif model_name=='svm':
        
        hyperparams = config['training']['svm']
        print('Training SVM with the following hyperparameters', hyperparams)
        feats = [x for x in df_train.columns if x not in ['Label', 'lat', 'lon','time', 'U850_UBOT', 'V850_VBOT']]
        X,y = get_data(df_train, feats, model_name=model_name, is_test=False)
        X = normalise(X)
        clf = svm.SVC(kernel=hyperparams['kernel'], gamma=hyperparams['gamma'], degree=hyperparams['degree'], C=hyperparams['C']).fit(X, y)

        X_test = get_data(df_test_, feats, model_name=model_name, is_test=True)
        X_test = normalise(X_test)
        y_pred = clf.predict(X_test)
        df_test['Label'] = y_pred
        df_submit = df_test[['SNo', 'Label']]
        df_submit.to_csv(os.path.join(config['test']['save_dir'], 'result.csv'), index=False)

    elif model_name=='xgboost':
        hyperparams = config['training']['xgboost']
        print('Training XGboost with the following hyperparameters', hyperparams)
        feats = [x for x in df_train.columns if x not in ['time', 'lat', 'lon','Label']]
        xgb_model = xgb.sklearn.XGBClassifier(
            learning_rate = hyperparams['learning_rate'],
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            min_child_weight=hyperparams['min_child_weight'],
            gamma=hyperparams['gamma'],
            subsample=hyperparams['subsample'],
            colsample_bytree=hyperparams['colsample_bytree'], 
            reg_alpha=hyperparams['reg_alpha'],
            reg_lambda = hyperparams['reg_lambda'],
            objective= 'multi:softmax',
            nthread=4,
            num_class=3,
            seed=42)
        y_pred = modelfit(xgb_model, df_train, feats, X_test=df_test_)
        df_test['Label'] = y_pred
        df_submit = df_test[['SNo', 'Label']]
        df_submit.to_csv(os.path.join(config['test']['save_dir'], 'result.csv'), index=False)

    else:
        raise ValueError('Not a valid model name. Available models: logistic_regression, svm, xgboost')
