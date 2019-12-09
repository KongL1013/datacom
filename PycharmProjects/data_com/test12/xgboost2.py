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


# In[8]:


print(len(data1),len(data))


# In[9]:


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


# In[10]:


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


# In[13]:


data=data.drop(['follow_topic','inter_topic','topic','title_t1','title_t2','desc_t1','desc_t2'],axis=1)


# In[40]:


#data=data.drop(['topic_sim0_max','topic_sim0_min','topic_sim0_avg','topic_sim1_max','topic_sim1_min','topic_sim1_avg'],axis=1)


# In[41]:


#data = data.drop(['topic_gz', 'topic_int', 't_invi', 't_quest', 'desc_quest_w', 'desc_quest_sw', 'desc_tit_w', 'desc_tit_sw', 'topic_quest'], axis=1)

# 缺省值处理

# ## 模型训练

# In[18]:


print(len(data)-1141683)


# In[19]:


from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
# 划分训练集和测试集
# y_train = data[:train.shape[0]]['label'].values
# X_train = data[:train.shape[0]].drop(['label'], axis=1).values

# X_test = data[train.shape[0]:].drop(['label'], axis=1).values
X = data[:2593669].drop(['label'], axis=1).values
y = data[:2593669]['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

parameters = {
    'max_depth': [10,15, 20, 25, 35],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'n_estimators': [1000,2000],
    'min_child_weight': [0, 2, 5, 10, 20, 50],
    'max_delta_step': [0, 0.2, 0.6, 1, 2],
    'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
}


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


model_xgboost = XGBClassifier(
                      # max_leaf_nodes=2**9-1,
                      max_depth=15,
                      learning_rate=0.01,
                      n_estimators=2000,
                      min_child_weight=5,
                      max_delta_step=0,
                      subsample=0.8,
                      colsample_bytree=0.7,
                      reg_alpha=0,
                      reg_lambda=0.4,
                      scale_pos_weight=0.8,
                      silent=False,
                      objective='binary:logistic',
                      missing=None,
                      eval_metric='auc',
                      seed=1440,
                      gamma=0,
                      n_jobs=-1
                      # nthread=40  #auto detect
)
# model_xgboost.fit(X_train,y_train)
gsearch = GridSearchCV(model_xgboost, param_grid=parameters, scoring='roc_auc', cv=5,n_jobs=-1)
gsearch.fit(X_train, y_train)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


y_pred = gsearch.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = roc_auc_score(y_test,predictions)
print("FINALL Accuracy: %.2f%%" % (accuracy * 100.0))



X_evaluate = data[2593669:].drop(['label'], axis=1).values

# y_pred = model_lgb.predict_proba(X_evaluate)

pd.DataFrame(y_pred[:, 1], columns=['y_pred'])

test = pd.read_csv('./invite_info_evaluate_1_0926.txt', header=None, sep='\t')
test.columns = ['问题id', '用户id', '邀请创建时间']
print(len(test))
# 用于保存提交结果
result_append = test[['问题id', '用户id', '邀请创建时间']]
result_append['Score'] = y_pred[:, 1]
print(result_append.head())
result_append.to_csv('result.txt', header=False, index=False, sep='\t')


