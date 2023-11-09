# Extreme-Weather-Events-Classification

This is the code for Kaggle competition which is a part of IFT6390 Course at Mila and aims to build a machine learning model for the classification of extreme weather conditions based on time point and location, latitude and longitude into 3 classes - Standard background conditions, Tropical cyclone and Atmospheric river. The training set contains 44,760 data points from 1996 to 2009, and the test set contains 10,320 data points from 2010 to 2013. Each data point consists of 16 atmospheric variables such as pressure, temperature and humidity, besides the latitude, longitude and time.

## Enviroment Setup
### Create virtual environemnt
```
conda create --name extreme_weather python
conda activate extreme_weather
```

### Install all libraries

```
pip3 install -r requirements.txt
```

## Setting up config parameters
All the hyperparameters for this project will be controlled by the `config.yaml` file.

### Data Parameters
These can be found in `config['dataset']` and need to be carefully set to read the data correctly.

  - `train_csv`: `<PATH_TO_THE_TRAINING_SET>`
  - `test_csv`: `<PATH_TO_THE_TEST_SET>`
  - `add_sq_feats`:  Set to `True` if you want to add the squared features to the dataset
  - `add_additional_feats`: Set to `True` if you want to add the additional features to the dataset. These are: ` 'U850\_V850', 'UBOT\_VBOT', 'PS\_PSL', 'T200\_T500','Z1000\_Z200', 'U850\_UBOT', 'V850\_VBOT'`

### Model Parameters
The model to use can be selected using `config['training']['model_name']` parameter and the options are- `logistic_regression`, `svm`, `xgboost`. Other model hyperparameters need to be carefully selected  as directed below.

## Models
### Available Models
There are 3 available models that can be trained on the given data. The hyperparameters for logistic regression gave the highest average scored model for the test dataset provided. The models are are:
  - Logistic Regression (Self-implemented)
  - SVM (Polynomial Kernel)
  - XGboost

### Setting up Logistic Regression
Set the `config['training']['model_name']=logistic_regression`
There a few options for running the logistic regression model which can be controlled using `config.yaml` file. Here are the hyperparameters for the model in  `config['training']['logistic_regression']`:

  - `is_cv`: Set to `True`, if you wanna run cross validation on the dataset to find the best parameters
  - `is_training_curve`: Set to `True`, if you wanna plot the loss and accuracy curves. This will split the training data to (train,val), save the training plots and retrain model using entire data and save results for test data
  - `is_weighted_loss`: Set `True` if you wanna use weighted cross entropy loss
  - `num_epochs`: Number of epochs you wanna train the model
  - `learning_rate`: Learning rate for gradient descent
  - `lambda1`: Regularization term for L1
  - `lambda2`: Regularization term for L2

### Setting up SVM
Set the `config['training']['model_name']=svm` and set the following hyperparameters in `config['training']['svm']`:

  - `kernel`: Kernel function to use
  - `gamma` : the kernel coefficient to use
  - `degree`: Degree of polynomial to use
  - `C`: Regularization parameter for squared L2 loss

### Setting up XGboost
Set the `config['training']['model_name']=xgboost` and set the following hyperparameters in `config['training']['xgboost']`

  - `learning_rate`: Learning rate for model
  - `n_estimators`: Number of estimators to run
  - `max_depth`: Max depth of tree
  - `min_child_weight`: Minimum weight of child
  - `gamma`:  the kernel coefficient to use
  - `subsample`: Subsample param to use
  - `colsample_bytree`: Col sample paramto use
  - `reg_alpha`: Regularization term for L1
  - `reg_lambda`: Regularization term for L2

Please note  that you can optimize the hyperparameters using the `./models/XGboost_finetuning.ipynb` and use those here.


## Model Training
Once everything is set run the following command:

```python3 train.py --config_path=./config.yaml```
