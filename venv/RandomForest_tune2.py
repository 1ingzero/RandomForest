import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time

start =time.clock()
# load data
train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem899/CYP2C19_PaDEL12_pcfp_train.csv')
target = 'label'
IDcol = 'Name'
predictors = [x for x in train.columns if x not in [target, IDcol]]

train_X = train[predictors]
train_Y = train[target]

#Grid search
param_grid = {
    'n_estimators':[1800, 2000, 2200],
    'min_samples_split':[ 8,10,12],
    'min_samples_leaf' :[3,4,5],
    'max_features':[40,45,50],
    'max_depth': [20, 30, 40, 50],
    'bootstrap': [True]}


# Create the random grid
if __name__=='__main__':

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier(random_state = 1)
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = GridSearchCV(estimator=rf, param_grid=param_grid,
                               scoring='roc_auc',
                              cv = 3, verbose=1,  n_jobs=-1,
                              )

    # Fit the random search model
    rf_random.fit(train_X, train_Y)
    print("Tune1 score:",rf_random.best_score_)
    print("Tune1 parameters set:", rf_random.best_params_)
elapsed = time.clock()-start
Time = elapsed/60
print("Used Time:",'%.1f'%Time,"min")
