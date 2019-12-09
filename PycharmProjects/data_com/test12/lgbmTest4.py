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


# In[14]:


data.head(2)


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


# In[21]:


model_lgb = LGBMClassifier(boosting_type='gbdt',
                           task='train',
                           num_leaves=2**9-1,
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
model_lgb.fit(X_train,
              y_train,
              eval_names=['train'],
              eval_metric={'auc'},
              eval_set=[(X_train, y_train),(X_test, y_test)],#, (X_test, y_test)
              #categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
              early_stopping_rounds=100)


# In[22]:


X_evaluate = data[2593669:].drop(['label'], axis=1).values


# In[23]:


y_pred = model_lgb.predict_proba(X_evaluate)


# [1]	train's auc: 0.763875	train's binary_logloss: 0.465954	valid_1's auc: 0.763206	valid_1's binary_logloss: 0.46561
# Training until validation scores don't improve for 50 rounds
# [2]	train's auc: 0.769659	train's binary_logloss: 0.464518	valid_1's auc: 0.769093	valid_1's binary_logloss: 0.464178
# [3]	train's auc: 0.77108	train's binary_logloss: 0.463213	valid_1's auc: 0.770482	valid_1's binary_logloss: 0.462878
# [4]	train's auc: 0.771961	train's binary_logloss: 0.461849	valid_1's auc: 0.771291	valid_1's binary_logloss: 0.461519
# [5]	train's auc: 0.773232	train's binary_logloss: 0.460513	valid_1's auc: 0.772525	valid_1's binary_logloss: 0.460188
# [6]	train's auc: 0.773901	train's binary_logloss: 0.459217	valid_1's auc: 0.773155	valid_1's binary_logloss: 0.458898
# [7]	train's auc: 0.774043	train's binary_logloss: 0.457969	valid_1's auc: 0.77331	valid_1's binary_logloss: 0.457653
# [8]	train's auc: 0.774356	train's binary_logloss: 0.456738	valid_1's auc: 0.773593	valid_1's binary_logloss: 0.456426
# [9]	train's auc: 0.774624	train's binary_logloss: 0.455526	valid_1's auc: 0.773802	valid_1's binary_logloss: 0.455221
# [10]	train's auc: 0.774838	train's binary_logloss: 0.454347	valid_1's auc: 0.773992	valid_1's binary_logloss: 0.454047
# 
# [1]	train's auc: 0.763775	train's binary_logloss: 0.46588	valid_1's auc: 0.763507	valid_1's binary_logloss: 0.465891
# Training until validation scores don't improve for 50 rounds
# [2]	train's auc: 0.769223	train's binary_logloss: 0.464457	valid_1's auc: 0.768826	valid_1's binary_logloss: 0.464472
# [3]	train's auc: 0.770967	train's binary_logloss: 0.463157	valid_1's auc: 0.770461	valid_1's binary_logloss: 0.463174
# [4]	train's auc: 0.772809	train's binary_logloss: 0.461796	valid_1's auc: 0.772294	valid_1's binary_logloss: 0.461816
# [5]	train's auc: 0.773698	train's binary_logloss: 0.460459	valid_1's auc: 0.773271	valid_1's binary_logloss: 0.460482
# [6]	train's auc: 0.773922	train's binary_logloss: 0.459161	valid_1's auc: 0.773457	valid_1's binary_logloss: 0.459188
# [7]	train's auc: 0.774355	train's binary_logloss: 0.457907	valid_1's auc: 0.773902	valid_1's binary_logloss: 0.457936
# [8]	train's auc: 0.774441	train's binary_logloss: 0.45667	valid_1's auc: 0.774017	valid_1's binary_logloss: 0.4567
# [9]	train's auc: 0.774521	train's binary_logloss: 0.455468	valid_1's auc: 0.774074	valid_1's binary_logloss: 0.455502
# [10]	train's auc: 0.77456	train's binary_logloss: 0.454296	valid_1's auc: 0.774088	valid_1's binary_logloss: 0.454333

# In[18]:


y_pred


# In[103]:


pd.DataFrame(y_pred[:, 1], columns=['y_pred'])


# In[24]:


test = pd.read_csv('./invite_info_evaluate_1_0926.txt', header=None, sep='\t')
test.columns = ['问题id', '用户id', '邀请创建时间']
print(len(test))
# 用于保存提交结果
result_append = test[['问题id', '用户id', '邀请创建时间']]


# In[25]:


result_append['Score'] = y_pred[:, 1]


# In[26]:


result_append.head()


# In[27]:


result_append.to_csv('result.txt', header=False, index=False, sep='\t')


# In[ ]:




