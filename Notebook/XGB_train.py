import re
import sys
import pandas as pd
import csv
import numpy as np 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV, KFold,cross_val_score
from sklearn.metrics import r2_score
import optuna
import sklearn
import pickle
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

prop = sys.argv[1]
scaled = False
optuna_trial= 0
if scaled :
    output = 'chalco_results2/'+ prop + '/' + 'scaled/'
else: 
    output = 'chalco_results2/'+ prop + '/' # + 'unscaled/'

print(prop)
import glob
folder = 'data_file_new/' 

# for file in glob.glob(folder):
for prop in [prop]:
    file = folder + prop+'.csv'
    print(file)
    df = pd.read_csv(file)
    if prop == 'TEC':
        df['TEC'] = np.log10(df['TEC'])
    data = df.values
    X_features = data[:,0:-1]
    Y_properties = data[:,-1]

    mean2 = Y_properties.mean()
#     scale = 10**int(np.log10(mean))
    print("mean2 -> " + str(mean2))
    
    Y_properties /= mean2
    mean = Y_properties.mean()
    std = Y_properties.std()
    Zs = (Y_properties-mean)/std
    mask = (Zs<3) & (Zs>-3)

    X_features = X_features[mask.ravel(),:]
    Y_properties = Y_properties[mask].ravel()
    
    X_train,X_test,Y_train,Y_test = train_test_split(X_features,Y_properties,test_size=0.2, random_state=33)
    
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    if scaled == True: 
        X_train_mean = X_train.mean()
        X_train_std = X_train.std() 
        X_train = (X_train - X_train_mean)/X_train_std
        X_test = (X_test - X_train_mean)/X_train_std 
    
    kfold = KFold(n_splits=4, shuffle=True,random_state=42)
    
    
    
    def objective(trial):
        if(df.shape[0] < 1000):
            list_trees = range(3,20,1)
        elif(1000<df.shape[0] < 5000):
            list_trees = range(3,50,1)
        else:
            list_trees = range(3, 75,1)
        param = {
             "random_state": trial.suggest_int("random_state", 1, 1000),
             "objective": "reg:squarederror",
             "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
             "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
             "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True), 
             "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),   # done for TEC so, change back to e-8
             "n_estimators" : trial.suggest_categorical('n_estimators', list_trees),
             "subsample": trial.suggest_float("subsample",0.7,1),
             "colsample_bytree": trial.suggest_float("colsample_bytree",0.7,1),
             "reg_alpha": trial.suggest_float("reg_alpha",1e-4,1, log=True),
             "reg_lambda": trial.suggest_float("reg_lambda",1e-4,1, log=True)
     
        }
        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 7)  #9
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        model = XGBRegressor(**param)  
        model.fit(X_train,Y_train,eval_set=[(X_test,Y_test)],early_stopping_rounds=100,verbose=False)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        return rmse
        
#         mean_cv_scorexg = cross_val_score(modelxg, X_train,Y_train, scoring='r2', 
#                                         cv=kfold, n_jobs=-1).mean()
#         return mean_cv_scorexg
        
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials= optuna_trial)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    param_xg = study.best_trial.params

    cvscores = []
    scores_train = []
    scores_test = []
#     for train, test in kfold.split(X_train, Y_train):
#         np.random.seed(5)
#         modelxg = XGBRegressor(**param_xg)
#         modelxg.fit(X_train[train],Y_train[train])
#         scores2 = r2_score(modelxg.predict(X_train[train]),Y_train[train])
#         scores = r2_score(modelxg.predict(X_train[test]),Y_train[test])
#         y_test_pred = modelxg.predict(X_test)
#         scores3 = r2_score(modelxg.predict(X_test),Y_test)
#         scores_train.append(scores2)
#         cvscores.append(scores)
#         scores_test.append(scores3)
    i=1
    for train, test in kfold.split(X_train, Y_train):
        np.random.seed(5)
        dftrain = pd.DataFrame(columns=['y_pr','y_ac'])
        dftest = pd.DataFrame(columns=['y_pr','y_ac'])
        modelxg = XGBRegressor(**param_xg)
        dftrain['y_ac']= Y_train[train] * mean2
        dftest['y_ac']= Y_train[test] * mean2
        
        modelxg.fit(X_train[train],Y_train[train])
        dftrain['y_pr']=modelxg.predict(X_train[train]) * mean2
        dftest['y_pr']= modelxg.predict(X_train[test]) * mean2
        scores2 = r2_score(modelxg.predict(X_train[train]),Y_train[train])
        scores = r2_score(modelxg.predict(X_train[test]),Y_train[test])
        y_test_pred = modelxg.predict(X_test)
        scores3 = r2_score(modelxg.predict(X_test),Y_test)
        scores_train.append(scores2)
        cvscores.append(scores)
        scores_test.append(scores3)
        dftrain.to_csv(output + prop + str(i)+'_train2'+'.csv')
        dftest.to_csv(output + prop + str(i) +'_test2'+'.csv')
        i +=1

    #file = re.sub('\.csv$', '', filename)
    pickle.dump([scores_train, cvscores, scores_test, param_xg,X_train_mean,X_train_std],open(output + prop+'_XGB2'+'.pickle','wb+'))
    print(np.array(scores_train))
    print(np.array(cvscores))
    print(np.array(scores_test))
    print(param_xg)