import numpy as np
import pandas as pd

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

def cv_svm(X_train, y_train, X_val, y_val):
    #RBF kernel
    # c_list = 
    pass
            