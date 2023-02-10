# This is a training file of Random Forestwhich inputs a processed data and used kfold technique for training and validating the scores. we have used optuna for hyperparameter optimisation.  we have taken parameters and for each property we have separately ran the training file with different parameter 

import torch
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.base import clone
import re
from keras.layers import Dropout
import sys
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from ipykernel import kernelapp as app
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import optuna
import sklearn
from sklearn import datasets
from sklearn.svm import SVR
import optuna
import shap
from sklearn.svm import SVR
import pickle
from sklearn.metrics import mean_squared_error
prop = sys.argv[1]
# epochss= int(sys.argv[2])
optuna_trial= int(sys.argv[2])

print(prop)
import glob

folder = 'processed_data/'
#t1 = time.time()
# for file in glob.glob(folder):
for prop in [prop]:
    file = folder + prop + '.csv'
    print(file)
    df = pd.read_csv(file)
    data = df.values
    X_features = data[:, 0:-1]
    Y_properties = data[:, -1]

    mean = Y_properties.mean()
    scale = 10 ** int(np.log10(mean))
    Y_properties /= scale
    mean = Y_properties.mean()
    std = Y_properties.std()
    Zs = (Y_properties - mean) / std
    mask = (Zs < 3) & (Zs > -3)

    X_features = X_features[mask.ravel(), :]
    Y_properties = Y_properties[mask].ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_properties, test_size=0.2, random_state=33)
    kfold = KFold(n_splits=4, shuffle=True, random_state=42)
    if df.shape[0] < 1000:
        list_trees = range(10, 100, 5)
    elif 1000 < df.shape[0] < 5000:
        list_trees = range(100, 300, 10)
    else:
        list_trees = range(300, 500, 10)


    def objective(trial):

        param = {
            'max_depth': trial.suggest_loguniform('max_depth', 2, 32),
            'n_estimators': trial.suggest_categorical('n_estimators', list_trees),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_uniform('max_features', 0.15, 1.0),
            'min_samples_split': trial.suggest_uniform('min_samples_split', 0.15, 1.0)
#             depending in property value
#             'min_impurity_split': trial.suggest_uniform('min_impurity_split',
        }
        model = RandomForestRegressor(**param)
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        return rmse


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=optuna_trial)
    params = study.best_trial.params
    params_rf = params

    cvscores_rf = []
    scores_train_rf = []
    scores_test_rf = []
    for train, test in kfold.split(X_train, Y_train):
        np.random.seed(5)
        model_rf = RandomForestRegressor(**params_rf)
        model_rf.fit(X_train[train], Y_train[train])
        scores2 = r2_score(model_rf.predict(X_train[train]), Y_train[train])
        scores = r2_score(model_rf.predict(X_train[test]), Y_train[test])
        y_test_pred = model_rf.predict(X_test)
        scores3 = r2_score(model_rf.predict(X_test), Y_test)

        scores_train_rf.append(scores2)
        cvscores_rf.append(scores)
        scores_test_rf.append(scores3)

    pickle.dump([scores_train_rf, cvscores_rf, scores_test_rf, params_rf], open(prop + '_RF' + '.pickle', 'wb+'))
    print(np.array(scores_train_rf))
    print(np.array(cvscores_rf))
    print(np.array(scores_test_rf))
    print(params_rf)
