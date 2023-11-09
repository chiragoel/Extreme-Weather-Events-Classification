import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

from models.logistic_regression import MultiClassLogisticRegression

def cv_logistic_regression(X_train, y_train, X_val, y_val):
    num_epochs_list = list(range(1000,10001,1000))
    hyperparam_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 0.9]

    best_acc=0.0
    best_num_epochs, best_lr, best_lambda1, best_lambda2 = 0.0, 0.0, 0.0, 0.0
    for  epoch in num_epochs_list:
        for learning_rate in hyperparam_list:
            for lambda1 in hyperparam_list:
                for lambda2 in hyperparam_list:
                    clf = MultiClassLogisticRegression(
                        is_weighted=True, 
                        learning_rate=learning_rate, 
                        lambda1=lambda1, 
                        lambda2=lambda2, 
                        num_epochs=epoch
                        )
                    W, b = clf.fit(X_train.T, y_train, X_val.T, y_val, verbose=0)
                    acc, loss = clf.evaluate_model(X_val.T, y_val, W, b)
                    if acc>best_acc:
                        best_num_epochs, best_lr, best_lambda1, best_lambda2 = epoch, learning_rate, lambda1, lambda2
                        best_acc=acc
                        
    print('Best hyparams: ', best_lr, best_lambda1, best_lambda2, best_num_epochs)
    return best_lr, best_lambda1, best_lambda2, best_num_epochs

def modelfit(alg, dtrain, predictors,sample_weights=None,X_test=None,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    '''
        Training the xgboost classifier
        Args:
            alg (xgboost obj): The initialised xgboost model object
            dtrain (pd.dataframe): The training dataset
            predictors (list): List of all the features to train the model on
            sample_weights (ndarray): Weights per sample assigned based on class frequency
            X_test (pd.dataframe): Test data for final prediction
            useTrainCV (bool): If use k-fold cross validation
            cv_folds (int): Number of folds for k-fold CV
            early_stopping_rounds (int): Number of iteration to wait before stopping to train when the loss doesn't change
    '''
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['Label'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0], cvresult)

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Label'])

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : {}".format(accuracy_score(dtrain['Label'].values, dtrain_predictions)))
    print(dtrain['Label'].values.shape, dtrain_predprob.shape, np.unique(dtrain_predictions, return_counts=True))

    #Test
    if len(X_test)>0:
        dtest_predictions = alg.predict(X_test[predictors])
        print('test: ', np.unique(dtest_predictions,return_counts=True))
    else:
        dtest_predictions=None

    # feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

    return dtest_predictions
            