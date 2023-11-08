import numpy as np
import pandas as pd

np.random.seed(0)


def compute_weighted_cross_entropy_loss(Y,Y_hat, weights=None):
  '''
    Compute the cross entropy loss
    Args:
        Y (ndarray: num_classes*num_samples): True labels - one-hot encoded
        Y_hat (ndarray: num_classes*num_samples): Predicted labels - one-hot encoded
        weights (ndarray: num_classes*1): Per class weight 
  '''
  m = Y.shape[1] #number of examples
  if weights is None:
    return (-1./m)*(np.sum(np.multiply(Y,np.log(Y_hat))))
  return (-1./m)*(np.sum(weights*np.multiply(Y,np.log(Y_hat))))

class MultiClassLogisticRegression:
    def __init__(self, is_weighted, learning_rate=0.01, lambda1=0.01, lambda2=0.01, num_epochs=3000, num_classes=3):
        '''
            Self implementation of Logistic Regression classifier
            Args:
                is_weighted (bool): If use weighted cross entropy loss for class imbalance
                learning_rate (float): Learning rate for classifer training
                lambda1 (float): Regularizer parameter for L1 regularization
                lambda2 (float): Regularizer parameter for L2 regularization
                num_epochs (int): Number of epochs for classifier training
                num_classes (int): Number of classes  available
        '''
        self.is_weighted = is_weighted
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.num_epochs = num_epochs
        self.num_classes = num_classes

    def _get_class_weights(self,y):
        '''
            Function that returns the weights for each class based on the training dataset
            Args:
                y (ndarray: num_classes*num_samples): True labels for all the samples in the training dataset - one hot encoded
            Returns: 
                class_weights (ndarray: num_classes*1): Array of computed class weights
        '''
        y = np.argmax(y, axis=1)
        class_weights = np.zeros((self.num_classes))

        for label in np.unique(y):
            class_weights[int(label)] = len(y[y!=label])/len(y)
        
        return class_weights.reshape(-1,1)

    def _compute_gradients(self, X, Y_hat, Y, class_weights):
        '''
            Gradient computation for gradient descent
            Args:
                X (ndarray: num_feats*num_samples):
                Y_hat (ndarray: num_classes*num_samples):
                Y (ndarray: num_classes*num_samples):
                class_weights (ndarray: num_classes*1): 
        '''

        if self.is_weighted:
            dZ = class_weights.reshape(-1,1)*(Y_hat-Y.T)
        else:
            dZ = (Y_hat-Y.T)
        
        dW = np.dot(X,dZ.T) #n_featsxnum_classes
        db = np.sum(dZ,axis = 1, keepdims = True) #num_classesx1

        return dZ, dW, db

    def softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def compute_pred(self, weights, bias, X):
        Z = weights.T.dot(X) + bias
        return self.softmax(Z)

    def evaluate_model(self, X_test, y_test, W, b):

        Y_hat = self.compute_pred(weights=W, bias=b, X=X_test)

        if self.is_weighted:
            loss = compute_weighted_cross_entropy_loss(y_test.T, Y_hat, weights=self.class_weights)
        else:
            loss = compute_weighted_cross_entropy_loss(y_test.T,Y_hat)

        loss+= - np.sum(self.lambda2*(W**2))/self.m - np.sum(self.lambda1*(np.abs(W)))/self.m

        y_pred = np.argmax(Y_hat, axis=0)
        y_true = np.argmax(y_test, axis=1)

        acc = len(y_pred[y_pred==y_true])/len(y_true)

        return acc, loss

    def fit(self, X_train, y_train, X_val, y_val, verbose=1):

        self.W = np.random.randn(X_train.shape[0],self.num_classes)*0.01 #16x3
        self.b = np.zeros((self.num_classes,1)) #3x1
        self.m  = X_train.shape[1]
        self.indexes = np.arange(X_train.shape[1])
        self.rows, self.columns = X_train.shape

        if self.is_weighted:
            self.class_weights = self._get_class_weights(y_train)

        best_val_loss = float('-inf')
        self.best_W, self.best_b = None, None
        for epoch in range(1,1+self.num_epochs):
            np.random.shuffle(self.indexes)
            X_train = X_train[:,self.indexes]
            X_train = X_train.reshape((self.rows, self.columns))
            y_train = y_train[self.indexes,:]
            y_train = y_train.reshape((self.columns, self.num_classes))

            Y_hat = self.compute_pred(weights=self.W, bias=self.b, X=X_train)
            if self.is_weighted:
                loss = compute_weighted_cross_entropy_loss(y_train.T, Y_hat, weights=self.class_weights)
            else:
                loss = compute_weighted_cross_entropy_loss(y_train.T,Y_hat)
            loss+= - np.sum(self.lambda2*(self.W**2))/self.m - np.sum(self.lambda1*(np.abs(self.W)))/self.m

            val_acc, val_loss = self.evaluate_model(X_val, y_val, self.W, self.b)

            if loss==np.nan:
                break
            if val_acc >= best_val_loss:
                self.best_W, self.best_b = self.W, self.b
                best_val_loss = val_acc

            dZ, dW, db = self._compute_gradients(X=X_train, Y_hat=Y_hat, Y=y_train, class_weights=self.class_weights)

            self.W = self.W - (1./self.m)*self.learning_rate*dW + (1./self.m)*self.lambda2*2.0*self.W + (1./self.m)*self.lambda1
            self.b = self.b - (1./self.m)*self.learning_rate*db

            if epoch%5==0 and verbose==1:
                print(f'Epoch {epoch}, Training loss {loss}, Val Loss {val_loss}, Val Acc {val_acc}')

        return self.best_W, self.best_b

