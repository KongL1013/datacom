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



from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
# 划分训练集和测试集
# y_train = data[:train.shape[0]]['label'].values
# X_train = data[:train.shape[0]].drop(['label'], axis=1).values

# X_test = data[train.shape[0]:].drop(['label'], axis=1).values
X = data[:2593669].drop(['label'], axis=1).values
X_xg_evaluate = data[2593669:].drop(['label'], axis=1).values

print("load model from file")
loaded_model = pickle.load(open("model_xgboost.pickle.dat", "rb"))
total_xg_pred = loaded_model.predict_proba(X)

##start
y_xg_pred = loaded_model.predict_proba(X_xg_evaluate)

get = np.r_[total_xg_pred,y_xg_pred]
print(get)
print("len get = ",len(get))
print("len data =",len(data))
# y_xg_pred1 = pd.DataFrame(get,columns=['pred1'])
# y_xg_pred2 = pd.DataFrame(get,columns=['pred2'])

##end


total_xg_pred_1 = pd.DataFrame(get[:,1],columns=['pred1'])
total_xg_pred_2 = pd.DataFrame(get[:,1],columns=['pred2'])

print(total_xg_pred[:5])
data = pd.concat([data,total_xg_pred_1],axis=1)
data = pd.concat([data,total_xg_pred_2],axis=1)
print(data.head())

################################################################################

# print("load model from file")
# loaded_model = pickle.load(open("model_xgboost.pickle.dat", "rb"))
# total_xg_pred = loaded_model.predict_proba(X)
# total_xg_pred_1 = pd.DataFrame(total_xg_pred[:,1],columns=['pred1'])
# print(total_xg_pred[:5])


# data['pred1'] = pd.concat([data['pred1'],y_xg_pred1],axis=0)
# data['pred2'] = pd.concat([data['pred2'],y_xg_pred2],axis=0)
# print("after merge")
# print(data.head())
# print(data.tail())
###################################################################################################
X = data[:2593669].drop(['label'], axis=1).values
y = data[:2593669]['label'].values

print("X = ",X[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("start lgb merge")
model_lgb = pickle.load(open("model_lgb1203.pickle.dat", "rb"))

from sklearn.metrics import accuracy_score

print('START TO SAVE RESULT!!!!!!!!!!!!!')
y_pred_test = model_lgb.predict(X_test)
predictions = [round(value) for value in y_pred_test]
accuracy = accuracy_score(y_test, predictions)
print("FINALL TEST Accuracy: %.2f%%" % (accuracy * 100.0))


X_evaluate = data[2593669:].drop(['label'], axis=1).values
y_pred = model_lgb.predict_proba(X_evaluate)
print("y_pred")
print(y_pred[:5,:])


test = pd.read_csv('./invite_info_evaluate_1_0926.txt', header=None, sep='\t')
test.columns = ['问题id', '用户id', '邀请创建时间']
print(len(test))
# 用于保存提交结果
result_append = test[['问题id', '用户id', '邀请创建时间']]
result_append['Score'] = y_pred[:, 1]
print(result_append.head())
result_append.to_csv('result_fix.txt', header=False, index=False, sep='\t')


