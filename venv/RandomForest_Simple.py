import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics

def SPSEQMCC(label, prediction):
    TP, FN, TN, FP = 0, 0, 0, 0
    for i in range(len(label)):
        if label[i] == 0:
            if prediction[i] == 0:
                TN = TN + 1
            else:
                FP = FP + 1
        else:
            if prediction[i] == 0:
                FN = FN + 1
            else:
                TP = TP + 1
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    print("TP,FN,TN,FP:", TP, FN, TN, FP)
    print("ACC：", ACC)
    print("SE：", SE)
    print("SP：", SP)
    L = (TN + FN) * (TN + FP) * (TP + FN) * (TP + FP)
    if L == 0:
        print("No MCC")
    else:
        MCC = (TP * TN - FP * FN) / (((TN + FN) * (TN + FP) * (TP + FN) * (TP + FP)) ** 0.5)
        print("MCC:", MCC)

train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem883/CYP2C9_PaDEL12_pcfp_train.csv')
test = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem883/CYP2C9_PaDEL12_pcfp_test.csv')
target = 'label'
IDcol = 'Name'
predictors = [x for x in train.columns if x not in [target, IDcol]]

train_X = train[predictors]
train_Y = train[target]

test_X = test[predictors]
test_Y = test[target]
rf0 = RandomForestClassifier(n_estimators =1000,min_samples_split=70,min_samples_leaf=20,max_depth=13,max_features='sqrt',
                             oob_score=True, random_state=1)
rf0.fit(train_X,train_Y)
print (rf0.oob_score_)
y_predprob_train = rf0.predict_proba(train_X)[:,1]
y_pred_train = rf0.predict(train_X)
print("AUC Score (Train): %f" % metrics.roc_auc_score(train_Y, y_predprob_train))
SPSEQMCC(train_Y,y_pred_train)

y_predprob_test = rf0.predict_proba(test_X)[:,1]
y_pred_test = rf0.predic(test_X)
print("AUC Score (Test): %f" % metrics.roc_auc_score(test_Y, y_predprob_test))
SPSEQMCC(test_Y,y_pred_test)

