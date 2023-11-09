import numpy as np
import pandas as pd

def split_dataset(data):
    train = data[[i for i in range(data.shape[0]) if i%5 in [0,1,2,3]]]
    val = data[[i for i in range(data.shape[0]) if i%5 in [4]]]
    return (train, val)

def normalise(X, norm='standard'):
    if norm=='standard':
        return (X-X.mean(axis=0))/X.std(axis=0)
    elif norm=='min_max':
        return (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    else:
        raise ValueError('Not a valid normalisation. Available options: standard, min_max')

def add_feats(df, dataset_config, model_name):
    '''
        Add the additional features to the dataframe
    '''
    feats = [x for x in df.columns if x not in ['Label', 'lat', 'lon','time']]
    if dataset_config['add_sq_feats']:
        df_squared = df[feats]**2
        df_squared.columns = [i+'_2' for i in list(df[feats].columns)]
        df_new = pd.concat([df, df_squared], axis=1)
        df = df_new
    
    if dataset_config['add_additional_feats']:
        df['PS_PSL'] = df['PS']*df['PSL']
        df['T200_T500'] = df['T200']*df['T500']
        df['Z1000_Z200'] = df['Z1000']*df['Z200']
        df['month'] = df['time'].apply(lambda x: int(str(x)[4:6]))
        df['U850_UBOT'] = df['U850']*df['UBOT']
        df['V850_VBOT'] = df['V850']*df['VBOT']
        df['U850_V850'] = df['U850']*df['V850']
        df['UBOT_VBOT'] = df['UBOT']*df['VBOT']
    
    return df

def convert_one_hot(y):
    
    num_classes = len(np.unique(y))
    y_one_hot = np.zeros((y.shape[0], num_classes))
    for i,  label in enumerate(y):
        y_one_hot[i][int(label)] = 1
    return y_one_hot

def get_data(df, features, model_name='logistic_regression', is_test=False):
    
    if is_test:
        return np.array(df[features])
    if model_name=='logistic_regression':
        X,y = np.array(df[features]), np.array(df['Label'])
        return X, convert_one_hot(y)
    elif model_name=='xgboost':
        pass
    elif model_name=='svm':
        X,y = np.array(df[features]), np.array(df['Label'])
        return X, y
    else:
        raise ValueError('Not a valid model name')


    