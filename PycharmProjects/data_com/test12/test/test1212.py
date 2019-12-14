#!/usr/bin/env python
# coding: utf-8

import lightgbm as lgb
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import gc

pd.set_option('display.max_columns', None)

with open('train20.pkl', 'rb') as file:
    data = pickle.load(file)
print(data.head())

X = data[:2593669].drop(['label'], axis=1).values
y = data[:2593669]['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("start xgboost")
model = XGBClassifier(
                max_depth=10,
                learning_rate=0.01,
                n_estimators=2000,
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
model.fit(X_train, y_train,
                  eval_metric='auc',
                  eval_set=[(X_train, y_train), (X_test, y_test)],  # , (X_test, y_test)
                  # categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
                  early_stopping_rounds=50)
cnt = 1212
print("save {} model!!!".format(cnt))
pickle.dump(model, open("model_xgboost{}.pickle.dat".format(cnt), "wb"))

print('START TO SAVE RESULT!!!!!!!!!!!!!')
y_pred_test = model.predict(X_test)
predictions = [round(value) for value in y_pred_test]
accuracy = accuracy_score(y_test, predictions)
print("FINALL TEST Accuracy: %.2f%%" % (accuracy * 100.0))

print('START TO SAVE RESULT!!!!!!!!!!!!!')
X_evaluate = data[2593669:].drop(['label'], axis=1).values
y_pred = model.predict_proba(X_evaluate)
print("y_pred")
print(y_pred[:5,:])


with open('test1.pkl', 'rb') as file:
    result_append = pickle.load(file)

print(data[2593669:].head())

result_append['Score'] = y_pred[:, 1]

print(result_append.head())

result_append.to_csv('result1212_xgboost.txt', header=False, index=False, sep='\t')
