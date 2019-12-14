#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import lightgbm as lgb
import pickle
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# import seaborn as sns
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
import gc
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_columns', None)

# with open('train20.pkl', 'rb') as file:
#     data = pickle.load(file)
# print(data.head())

# testdata = data[:1000]

# with open('test1213.pkl','rb') as file:
#     # pickle.dump(testdata,file)
#     data = pickle.load(file)
# # print(data.head())
# isnull_index = data['title_ans_t1_max'].isnull()
# # print(isnull_index)
# data_isnull = data[isnull_index]
# data_notnull = data[~isnull_index]
# print(len(data_notnull['title_ans_t1_max']))
# print(len(data_notnull.index))
# print(data_notnull['title_ans_t1_max'])
# print(data_isnull['title_ans_t1_max'])
#
# data_isnull.reset_index(drop=True)
# data_notnull = data_notnull.reset_index(drop=True)
# print(data_notnull['title_ans_t1_max'])
# print(data_isnull['title_ans_t1_max'])
# print(data_isnull.columns)
#
# with open('index.txt','w') as file:
#     # pickle.dump(data_isnull.columns,file)
#     for i in data_isnull.columns:
#         file.write(str(i)+'\t')

a = pd.Series(np.arange(1,2,0.1),index=None,name='str0.1')
b = pd.Series(np.arange(0,10),index=np.arange(10),name='kk')
c = pd.Series(np.arange(20,30),index=np.arange(10),name='big')

test = pd.DataFrame()
test = test.append(a)
test = test.append(b)
test = test.append(c)
test[2]['kk'] = np.nan
# print(test)
# print(test[2]['kk'])
# print(test.iloc[1,2])
# print(test[2])

def pp(d):
    print('ddddddd')
    print(d)
    # print(d[1])
    # print(pd.isnull(d[1]))
# ee = pd.isnull(test.iloc[1,2])
# test.apply(pp,axis=1)
# print(ee)
# dd = test.apply(lambda x:x if pd.isnull(x[1]) else x/2,axis=0)
# print(dd)
# index_null = test[2].isnull()
# select = test[index_null]
# print(select)

# test = test.drop(columns=[1,2],axis=1)
# print(test)

# data_notnull = data.drop(index=isnull_index,axis=0)
# print(len(data_notnull))
# unan =  data.head().apply(lambda x )


test2 = pd.DataFrame(np.random.rand(8, 4),columns=['a','b','c','d'])

test2['b'][2] = np.nan
test2['b'][3] = np.nan
test2['b'][4] = np.nan
test2['b'][5] = np.nan
test2['b'][6] = np.nan
print(test2)

index = test2['b'].isnull()
ee = test2[index]
print(ee)
select_index = np.random.choice(test2[index].index,3,replace=False)
print(np.sort(select_index))
data = test2.iloc[select_index,:]
print(data)
# print(data.iloc[1,:])
# print(data)
# print(data[data['b'].isnull()])
# print(select_index)
# print(test2.iloc[select_index,:])
# data1 = test2.drop(index=index,axis=0)
# data2 = test2.drop([0,2,3],axis=0)
# test2 = test2.reset_index()
# data1 = data1.reset_index()
# data2 = data2.reset_index()
# # print(data1)

# data = pd.concat([data1,data2],axis=0)
# # print(data)
# data = data.sort_values(by="index" , ascending=True)
# print(data)

# data1 = pd.DataFrame(np.arange(0,))
# print(test2[:2]['a'])
# test_isnull_index = test2['b'].isnull()
# print(test_isnull_index)
# test_isnull = test2[test_isnull_index]
# print(test_isnull)
# test_isnull = test_isnull.reindex(index=np.arange(len(test_isnull)))
# print(test_isnull)







