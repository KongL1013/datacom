#!/usr/bin/env python
# coding: utf-8

# In[1]:

import lightgbm as lgb
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import gc
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_columns', None)

with open('train20.pkl', 'rb') as file:
    data = pickle.load(file)
print(data.head())



# X = data[:2593669].drop(['label'], axis=1).values
X = data.drop(['label'], axis=1).values
print("load model from file")
loaded_model1 = pickle.load(open("all_xgboost1.pickle.dat", "rb"))
total_xg_pred1 = loaded_model1.predict_proba(X)
loaded_model2 = pickle.load(open("all_xgboost2.pickle.dat", "rb"))
total_xg_pred2 = loaded_model2.predict_proba(X)
loaded_model3 = pickle.load(open("all_xgboost3.pickle.dat", "rb"))
total_xg_pred3 = loaded_model3.predict_proba(X)
total_xg_pred_1 = pd.DataFrame(total_xg_pred1[:,1],columns=['xg_pred1'])
total_xg_pred_2 = pd.DataFrame(total_xg_pred2[:,1],columns=['xg_pred2'])
total_xg_pred_3 = pd.DataFrame(total_xg_pred3[:,1],columns=['xg_pred3'])

data = pd.concat([data,total_xg_pred_1],axis=1)
data = pd.concat([data,total_xg_pred_2],axis=1)
data = pd.concat([data,total_xg_pred_3],axis=1)
print(data.head())

print("start lgb merge")
loaded_model_bgm1 = pickle.load(open("all_xgboost4.pickle.dat", "rb"))
total_bgm_pred1 = loaded_model_bgm1.predict_proba(X)
loaded_model_bgm2 = pickle.load(open("all_xgboost5.pickle.dat", "rb"))
total_bgm_pred2 = loaded_model_bgm2.predict_proba(X)
total_bgm_pred_1 = pd.DataFrame(total_bgm_pred1[:,1],columns=['bgm_pred1'])
total_bgm_pred_2 = pd.DataFrame(total_bgm_pred2[:,1],columns=['bgm_pred2'])
data = pd.concat([data,total_bgm_pred_1],axis=1)
data = pd.concat([data,total_bgm_pred_2],axis=1)
print(data.head())




from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
# 划分训练集和测试集
# y_train = data[:train.shape[0]]['label'].values
# X_train = data[:train.shape[0]].drop(['label'], axis=1).values

# X_test = data[train.shape[0]:].drop(['label'], axis=1).values
X = data[:2593669].drop(['label'], axis=1).values
y = data[:2593669]['label'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import gc

from lightgbm import LGBMClassifier



from sklearn.model_selection import StratifiedKFold

n_fold = 5
steps = 2300
depth = 3


skf = StratifiedKFold(n_splits=n_fold, random_state=2020, shuffle=False)
X_evaluate = data[2593669:].drop(['label'], axis=1).values
y_pred = np.zeros([X_evaluate.shape[0],2])
cnt = 1



for index ,(train_index,test_index) in enumerate(skf.split(X , y)): #训练数据五折
    print("start train time = {cnt}".format(cnt=cnt))
    print("train_index = ",train_index)
    print("train_index_% = ", len(train_index)/len(X))
    train_x, test_x, train_y, test_y = X[train_index], X[test_index], y[
        train_index], y[test_index]
    print("start training")
    # if cnt%2 != 0:
    if True:
        print("model_xgboost")
        model = XGBClassifier(
            max_depth=depth,
            learning_rate=0.001,
            n_estimators=steps,
            min_child_weight=5,  # 5
            max_delta_step=0,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0,
            reg_lambda=0.4,
            scale_pos_weight=0.8,
            silent=True,
            objective='binary:logistic',
            missing=None,
            eval_metric='auc',
            seed=1440,
            gamma=0,
            n_jobs=-1
            # nthread=40
        )
        model.fit(train_x, train_y,
                          eval_metric='auc',
                          eval_set=[(train_x, train_y), (test_x, test_y)],  # , (X_test, y_test)
                          # categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
                          early_stopping_rounds=2000)
        print("save {} model!!!".format(cnt))
        pickle.dump(model, open("./model/mydata_model_xgboost{}.pickle.dat".format(cnt), "wb"))
    else:
        print("LGBMClassifier")
        model = LGBMClassifier(boosting_type='gbdt',
                               task='train',
                               num_leaves=2 ** depth - 1,
                               num_iterations=steps,
                               learning_rate=0.01,
                               n_estimators=2000,
                               max_bin=425,
                               subsample_for_bin=50000,
                               objective='binary',
                               min_split_gain=0,
                               min_child_weight=5,
                               min_child_samples=10,
                               feature_fraction=0.9,
                               feature_fraction_bynode=0.8,
                               drop_rate=0.05,
                               subsample=0.8,
                               subsample_freq=1,
                               colsample_bytree=1,
                               reg_alpha=3,
                               reg_lambda=5,
                               seed=1000,
                               # n_jobs=4,
                               silent=True
                               )
        # 建议使用CV的方式训练预测。
        model.fit(train_x,
                  train_y,
                  eval_names=['train'],
                  eval_metric={'auc'},
                  eval_set=[(train_x, train_y), (test_x, test_y)],  # , (X_test, y_test)
                  # categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
                  early_stopping_rounds=50)
        print("save {} model!!!".format(cnt))
        pickle.dump(model, open("./model/mydata_LGBMClassifier{}.pickle.dat".format(cnt), "wb"))

    gc.collect()  # 垃圾清理，内存清理

    y_pred_test = model.predict(test_x)
    predictions = [round(value) for value in y_pred_test]
    accuracy = accuracy_score(test_y, predictions)
    print("FINALL TEST Accuracy: %.2f%%" % (accuracy * 100.0))

    y_pred += model.predict_proba(X_evaluate)
    print('len X_evaluate =',len(X_evaluate))
    print("len y_pred=",len(y_pred))
    print("y_pred")
    print(y_pred[:5, :]/cnt)
    cnt += 1

y_pred = y_pred/n_fold


###sigle xgboost
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)



print("start xgboost")
model_xgboost = XGBClassifier(
                      max_depth=2,
                      learning_rate=0.001,
                      n_estimators=2200,
                      min_child_weight=5, #5
                      max_delta_step=0,
                      subsample=0.8,
                      colsample_bytree=0.7,
                      reg_alpha=0,
                      reg_lambda=0.4,
                      scale_pos_weight=0.8,
                      silent=True,
                      objective='binary:logistic',
                      missing=None,
                      eval_metric='auc',
                      seed=1440,
                      gamma=0,
                      n_jobs=-1
                      # nthread=40
)
model_xgboost.fit(X_train,y_train,
              eval_metric='auc',
              eval_set=[(X_train, y_train),(X_test, y_test)],#, (X_test, y_test)
              #categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
              early_stopping_rounds=200)
# save model to file
pickle.dump(model_xgboost, open("./model/model_xgboost.pickle2.dat", "wb"))


# model_xgboost = pickle.load(open("./model/model_xgboost.pickle.dat", "rb"))


y_pred_test = model_xgboost.predict(X_test)
predictions = [round(value) for value in y_pred_test]
accuracy = accuracy_score(y_test, predictions)
print("FINALL TEST Accuracy: %.2f%%" % (accuracy * 100.0))


X_evaluate = data[2593669:].drop(['label'], axis=1).values

y_pred = model_xgboost.predict_proba(X_evaluate)

'''
###sigle xgboost



with open('test1.pkl', 'rb') as file:
    result_append = pickle.load(file)

print(data[2593669:].head())

result_append['Score'] = y_pred[:, 1]

print(result_append.head())

result_append.to_csv('./model/result1213_mydata2.txt', header=False, index=False, sep='\t')

