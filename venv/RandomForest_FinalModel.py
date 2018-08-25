import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics

#定义模型评价函数
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
    print("ACC：", '%.3f'%ACC)
    print("SE：", '%.3f'%SE)
    print("SP：", '%.3f'%SP)
    L = (TN + FN) * (TN + FP) * (TP + FN) * (TP + FP)
    if L == 0:
        print("No MCC")
    else:
        MCC = (TP * TN - FP * FN) / (((TN + FN) * (TN + FP) * (TP + FN) * (TP + FP)) ** 0.5)
        print("MCC:", '%.3f'% MCC)

# 提取出特征值和标签
train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem891/CYP2D6_PaDEL12_pcfp_train.csv')
test = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem891/CYP2D6_PaDEL12_pcfp_test.csv')
target = 'label'
IDcol = 'Name'
predictors = [x for x in train.columns if x not in [target, IDcol]]

train_X = train[predictors]
train_Y = train[target]

test_X = test[predictors]
test_Y = test[target]

#构建随机森林模型
rf0 = RandomForestClassifier(n_estimators =2200,min_samples_split=3,min_samples_leaf=4,max_features=50,max_depth=50,
                             bootstrap=False, random_state=1)
#训练模型
rf0.fit(train_X,train_Y)

#得到模型在训练集上结果
y_predprob_train = rf0.predict_proba(train_X)[:,1]
y_pred_train = rf0.predict(train_X)
SPSEQMCC(train_Y,y_pred_train)
print("AUC Score (Train): %.3f" % metrics.roc_auc_score(train_Y, y_predprob_train))

#得到模型在测试集上结果
y_predprob_test = rf0.predict_proba(test_X)[:,1]
y_pred_test = rf0.predict(test_X)
SPSEQMCC(test_Y,y_pred_test)
print("AUC Score (Test): %.3f" % metrics.roc_auc_score(test_Y, y_predprob_test))

