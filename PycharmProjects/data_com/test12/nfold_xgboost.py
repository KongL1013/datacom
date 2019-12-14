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

X = data[:2593669].drop(['label'], axis=1).values
y = data[:2593669]['label'].values

n_fold = 5
steps = 2500

skf = StratifiedKFold(n_splits=n_fold, random_state=0, shuffle=False)
X_evaluate = data[2593669:].drop(['label'], axis=1).values
y_pred = np.zeros([X_evaluate.shape[0],2])
cnt = 1

for index ,(train_index,test_index) in enumerate(skf.split(X , y)): #训练数据五折
    print("start train time = {cnt}".format(cnt=cnt))
    print("train_index = ",train_index)
    print("train_index_% = ", len(train_index)/len(X))
    train_x, test_x, train_y, test_y = X[train_index], X[test_index], y[
        train_index], y[test_index]
    print("start train")
    if True:
        print("model_xgboost")
        model = XGBClassifier(
            max_depth=10,
            learning_rate=0.01,
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
                          early_stopping_rounds=50)
        print("save {} model!!!".format(cnt))
        pickle.dump(model, open("all_xgboost{}.pickle.dat".format(cnt), "wb"))
    else:
        print("LGBMClassifier")
        model = LGBMClassifier(boosting_type='gbdt',
                               task='train',
                               num_leaves=2 ** 9 - 1,
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
        pickle.dump(model, open("LGBM1212Classifier{}.pickle.dat".format(cnt), "wb"))

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


with open('test1.pkl', 'rb') as file:
    result_append = pickle.load(file)

print(data[2593669:].head())

result_append['Score'] = y_pred[:, 1]

print(result_append.head())

result_append.to_csv('all_xg_result1_nfold.txt', header=False, index=False, sep='\t')
