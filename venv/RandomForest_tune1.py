import numpy as np
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time

start = time.clock()
# load data
train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem899/CYP2C19_PaDEL12_pcfp_train.csv')
target = 'label'
IDcol = 'Name'
predictors = [x for x in train.columns if x not in [target, IDcol]]

train_X = train[predictors]
train_Y = train[target]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# Create the random grid
if __name__=='__main__':
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier(random_state = 1)
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='roc_auc',
                              cv = 3, random_state=1,verbose=1, n_jobs=-1,
                              )

    # Fit the random search model
    rf_random.fit(train_X, train_Y)
    print("Tune1 score:",rf_random.best_score_)
    print("Tune1 parameters set:", rf_random.best_params_)
    elapsed = time.clock()-start
    Time = elapsed / 60
    print("Used Time:", '%.1f' % Time, "min")