# This is a training file of SVM which inputs a processed data and used kfold technique for training and validating the scores. we have used optuna for hyperparameter optimisation.  we have taken parameters and for each property we have separately ran the training file with different parameter 

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
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV, KFold,cross_val_score
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
# # epochss= int(sys.argv[2])
optuna_trial= int(sys.argv[2])
# prop ="ND20"
# optuna_trial = 10
print(prop)
import glob
folder = 'processed_data/'
# for file in glob.glob(folder):
for prop in [prop]:
    file = folder+ prop+'.csv'
    print(file)
    df = pd.read_csv(file)
    data = df.values
    X_features = data[:,0:-1]
    Y_properties = data[:,-1]

    mean = Y_properties.mean()
    scale = 10**int(np.log10(mean))
    Y_properties /= scale
    mean = Y_properties.mean()
    std = Y_properties.std()
    Zs = (Y_properties-mean)/std
    mask = (Zs<3) & (Zs>-3)

    X_features = X_features[mask.ravel(),:]
    Y_properties = Y_properties[mask].ravel()
    
    X_train,X_test,Y_train,Y_test = train_test_split(X_features,Y_properties,test_size=0.2, random_state=33)
    kfold = KFold(n_splits=4, shuffle=True,random_state=42)
    
    def objective(trial):

        C = trial.suggest_uniform('C', 0.01, 10)
        gamma = trial.suggest_uniform('gamma', 0.0009,0.1)
        
        modelsv = SVR(kernel='rbf', C=C, gamma= gamma)
        modelsv.fit(X_train,Y_train)
        preds = modelsv.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials= optuna_trial)
    # print('Number of finished trials:', len(study.trials))
    # print('Best trial:', study.best_trial.params)
    params = study.best_trial.params
    params_svm = params
    
    cvscores = []
    scores_train = []
    scores_test = []
    for train, test in kfold.split(X_train, Y_train):
        np.random.seed(5)
        modelsv = SVR(**params)
        modelsv.fit(X_train[train],Y_train[train])
        scores2 = r2_score(modelsv.predict(X_train[train]),Y_train[train])
    
        scores = r2_score(modelsv.predict(X_train[test]),Y_train[test])
    
        y_test_pred = modelsv.predict(X_test)
        scores3 = r2_score(modelsv.predict(X_test),Y_test)
    
    #     print(model_rf.get_params)
        scores_train.append(scores2)
        cvscores.append(scores)
        scores_test.append(scores3)

    #filename = re.sub('\.csv$', '', filename)
    pickle.dump([scores_train, cvscores, scores_test, params_svm],open(prop+'_SVM'+'.pickle','wb+'))
    
    print(np.array(scores_train))
    print(np.array(cvscores))
    print(np.array(scores_test))
    print(params_svm)