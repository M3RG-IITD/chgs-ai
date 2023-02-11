from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, DotProduct
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV, KFold,cross_val_score
import math
import torch
torch.potrs = torch.cholesky_solve
import gpytorch
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import csv
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score as r2
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import optuna
import sys
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle



prop = sys.argv[1]
# epochss= int(sys.argv[2])
optuna_trial= int(sys.argv[2])
print(prop)
import glob
folder = 'data/'
# for file in glob.glob(folder):
for prop in [prop]:
    file = folder+  prop+'.csv'
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
    
    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_properties, test_size=0.2, random_state=42)
    # from numpy import arange
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)
    kf = KFold(n_splits=4,shuffle=True, random_state=42)



    def objective(trial):
        param = {
                'alpha': trial.suggest_loguniform('alpha', 1e-11, 1e-3),
                'kernel': trial.suggest_categorical('kernel',[RBF(l) for l in np.arange(0,2,0.01)]),
        }
        a = list(param.items())
        model = GaussianProcessRegressor(alpha=a[0][1],kernel = a[1][1] + DotProduct() + WhiteKernel())  
        model.fit(train_x,train_y)
        preds = model.predict(test_x)
        rmse = np.sqrt(mean_squared_error(test_y, preds))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials= optuna_trial)
    param_xg = study.best_trial.params
    param_xg['kernel'] += (DotProduct()+ WhiteKernel())

    cvscores = []
    scores_train = []
    scores_test = []
    X_train =train_x
    Y_train =train_y
    X_test =test_x
    Y_test =test_y
    for train, test in kf.split(X_train, Y_train):
        np.random.seed(5)
        modelsv = GaussianProcessRegressor(**param_xg)
        modelsv.fit(X_train[train],Y_train[train])
        scores2 = r2_score(modelsv.predict(X_train[train]),Y_train[train])
        scores = r2_score(modelsv.predict(X_train[test]),Y_train[test])
        y_test_pred = modelsv.predict(X_test)
        scores3 = r2_score(modelsv.predict(X_test),Y_test)
        scores_train.append(scores2)
        cvscores.append(scores)
        scores_test.append(scores3)
    
    #filename = re.sub('\.csv$', '', filename)
    pickle.dump([scores_train, cvscores, scores_test, param_xg],open(prop+'_Gaussian'+'.pickle','wb+'))
    
    print(np.array(scores_train))
    print(np.array(cvscores))
    print(np.array(scores_test))
    print(param_xg)