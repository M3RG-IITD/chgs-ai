# This is a training file of NN which inputs a processed data and used kfold technique for training and validating the scores. we have used optuna for hyperparameter optimisation.  we have taken parameters and for each property we have separately ran the training file with different parameter 

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
import time

prop = sys.argv[1]
epochss= int(sys.argv[2])
optuna_trial= int(sys.argv[3])

print(prop)
import glob
folder = 'density/'
t1 = time.time()
# for file in glob.glob(folder):
for prop in [prop]:
    file = prop+'.csv'
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
    
    def create_model(trial):
    
        n_layers = trial.suggest_int("n_layers", 2, 5)
        model = Sequential()
        for i in range(n_layers):
            num_hidden = trial.suggest_int("n_units_l{}".format(i), df.shape[1]/2,2*df.shape[1] ,1)
            kernel_initializer = trial.suggest_categorical('kernel_initializer',['he_uniform', 'random_normal'])
            activation = trial.suggest_categorical('activation', ["relu" , "tanh" , "LeakyReLU"])
            if activation == "LeakyReLU":
                alpha = trial.suggest_categorical('alpha', [0.1,0.2,0.3])
            model.add(Dense(num_hidden,input_dim=df.shape[1]-1, activation=activation, kernel_initializer = kernel_initializer))
            dropout = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.3)
            model.add(Dropout(rate=dropout))
        model.add(Dense(1))

        # We compile our model with a sampled learning rate.
        lr = trial.suggest_uniform("lr", 1e-5, 1e-1)
        opt =  Adam(learning_rate=lr)
        model.compile(
            loss='mean_squared_error',
            optimizer=opt,
            metrics=["accuracy"],
        )
        return model

    x_train,x_valid ,y_train,y_valid  = train_test_split(X_train,Y_train,test_size=0.2, random_state=42)
    def objective(trial):

        keras.backend.clear_session()
        model = create_model(trial)
        model.fit(
            x_train,
            y_train,
            batch_size= 256,
            epochs= epochss,
            validation_data=(x_valid, y_valid),
            verbose=1,
        )
        score = r2_score(y_valid, model.predict(x_valid))
        return score


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trial)
    paramsnn = study.best_trial.params
    a = list(paramsnn.items())
    cvscores = []
    scores_train = []
    scores_test = []
    for train, test in kfold.split(X_train, Y_train):
        n_layers = a[0][1]
        model2 = Sequential()
        model2.add(Dense(a[1][1], input_dim=df.shape[1]-1, kernel_initializer=a[2][1], activation=a[3][1]))
        model2.add(Dropout(rate=a[4][1]))
        index=5
        for i in range(n_layers-1):
            model2.add(Dense(a[index][1],input_dim=df.shape[1]-1, activation="relu", kernel_initializer = a[2][1]))
            model2.add(Dropout(rate=a[index+1][1]))
            index = index+2
        model2.add(Dense(1))
        opt =  Adam(learning_rate=a[-1][1])
        model2.compile(loss='mean_squared_error', optimizer=opt,metrics=["accuracy"])
        model2.fit(X_train[train],Y_train[train],epochs=50)
        scores2 = r2_score(model2.predict(X_train[train]),Y_train[train])
        scores = r2_score(model2.predict(X_train[test]),Y_train[test])
        scores3 = r2_score(model2.predict(X_test),Y_test)


        scores_train.append(scores2)
        cvscores.append(scores)
        scores_test.append(scores3)
    #filename = re.sub('\.csv$', '', filename)
    pickle.dump([scores_train, cvscores, scores_test, paramsnn],open(file+'_NN'+'.pickle','wb+'))
    print(scores_train)
    print(cvscores)
    print(scores_test)