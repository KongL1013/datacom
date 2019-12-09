#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lightgbm as lgb
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


with open('train11.pkl', 'rb') as file:
    data = pickle.load(file)


# In[4]:


def fill_null(data, col, null_value):
    use_avg = data[data[col] != null_value][col].mean()
    data.loc[data[col] == null_value, col] = use_avg
    return data
def fill_null_nan(data, col, null_value):
    use_avg = np.nan
    data.loc[data[col] == null_value, col] = use_avg
    return data


# In[5]:


with open('train10.pkl', 'rb') as file:
    data1 = pickle.load(file)


print(len(data1),len(data))



data['topic_sim0_max']=data1['topic_sim0'].apply(lambda x:x[0])
data['topic_sim0_avg']=data1['topic_sim0'].apply(lambda x:x[1])
data['topic_sim0_min']=data1['topic_sim0'].apply(lambda x:x[2])
data['topic_sim0_std']=data1['topic_sim0'].apply(lambda x:x[3])
data['topic_sim0_num']=data1['topic_sim0'].apply(lambda x:x[4])

data['topic_sim1_max']=data1['topic_sim1'].apply(lambda x:x[0])
data['topic_sim1_avg']=data1['topic_sim1'].apply(lambda x:x[1])
data['topic_sim1_min']=data1['topic_sim1'].apply(lambda x:x[2])
data['topic_sim1_std']=data1['topic_sim1'].apply(lambda x:x[3])
data['topic_sim1_num']=data1['topic_sim1'].apply(lambda x:x[4])
data['topic_sim1_max1']=data1['topic_sim1'].apply(lambda x:x[5])
data['topic_sim1_min1']=data1['topic_sim1'].apply(lambda x:x[6])



fill_null(data, 'topic_sim0_max', -2)
fill_null(data, 'topic_sim0_avg', -2)
fill_null(data, 'topic_sim0_min', -2)
fill_null(data, 'topic_sim0_std', -2)
fill_null(data, 'topic_sim0_num', -2)
fill_null(data, 'topic_sim1_max', -2)
fill_null(data, 'topic_sim1_avg', -2)
fill_null(data, 'topic_sim1_min', -2)
fill_null(data, 'topic_sim1_std', -2)
fill_null(data, 'topic_sim1_num', -2)
fill_null(data, 'topic_sim1_max1', -2)
fill_null(data, 'topic_sim1_min1', -2)


data=data.drop(['follow_topic','inter_topic','topic','title_t1','title_t2','desc_t1','desc_t2'],axis=1)


print(len(data)-1141683)

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
    print("start xgboost")
    if cnt%2 != 0:
        print("model_xgboost")
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
        model.fit(train_x, train_y,
                          eval_metric='auc',
                          eval_set=[(train_x, train_y), (test_x, test_y)],  # , (X_test, y_test)
                          # categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
                          early_stopping_rounds=50)
        print("save {} model!!!".format(cnt))
        pickle.dump(model, open("model_xgboost{}.pickle.dat".format(cnt), "wb"))
    else:
        print("LGBMClassifier")
        model = LGBMClassifier(boosting_type='gbdt',
                                   task='train',
                                   num_leaves=2 ** 9 - 1,
                                   num_iterations=2000,
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
                                   n_jobs=4,
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
        pickle.dump(model, open("LGBMClassifier{}.pickle.dat".format(cnt), "wb"))

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


# print("start xgboost")
# model_xgboost = XGBClassifier(
#                       max_depth=10,
#                       learning_rate=0.01,
#                       n_estimators=2500,
#                       min_child_weight=5, #5
#                       max_delta_step=0,
#                       subsample=0.8,
#                       colsample_bytree=0.7,
#                       reg_alpha=0,
#                       reg_lambda=0.4,
#                       scale_pos_weight=0.8,
#                       silent=True,
#                       objective='binary:logistic',
#                       missing=None,
#                       eval_metric='auc',
#                       seed=1440,
#                       gamma=0,
#                       n_jobs=-1
#                       # nthread=40
# )
# model_xgboost.fit(X_train,y_train,
#               eval_metric='auc',
#               eval_set=[(X_train, y_train),(X_test, y_test)],#, (X_test, y_test)
#               #categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
#               early_stopping_rounds=50)



# save model to file
# print("save model!!!")
# pickle.dump(model_xgboost, open("model_xgboost1208.pickle.dat", "wb"))
#
#
# print('START TO SAVE RESULT!!!!!!!!!!!!!')
# y_pred_test = model_xgboost.predict(X_test)
# predictions = [round(value) for value in y_pred_test]
# accuracy = accuracy_score(y_test, predictions)
# print("FINALL TEST Accuracy: %.2f%%" % (accuracy * 100.0))
#
#
# X_evaluate = data[2593669:].drop(['label'], axis=1).values
# y_pred = model_xgboost.predict_proba(X_evaluate)
# print("y_pred")
# print(y_pred[:5,:])
#


test = pd.read_csv('./invite_info_evaluate_1_0926.txt', header=None, sep='\t')
test.columns = ['问题id', '用户id', '邀请创建时间']
print(len(test))
# 用于保存提交结果
result_append = test[['问题id', '用户id', '邀请创建时间']]
result_append['Score'] = y_pred[:, 1]
print(result_append.head())
result_append.to_csv('result_xg_lg.txt', header=False, index=False, sep='\t')

