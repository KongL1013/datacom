import pickle
import numpy as np
import pandas as pd


# data1 = pd.read_csv('result_xgboost.txt', header=None, sep='\t')
data2 = pd.read_csv('result1.txt', header=None, sep='\t')
# print(data1.head())
# print(data2.head())
# data1[3] = (data1[3]+data2[3])/2
# print(data1.head())
#
#
# data1.to_csv('result.txt', header=False, index=False, sep='\t')
test = data2.head(5)
test.columns = ['q','w','e','r']
test =pd.concat( [test,pd.DataFrame(np.arange(8),columns=['w'])])
print(test)
test['w'] =np.arange(5)
data = np.arange(9.0,step = 1.2)
tt = pd.DataFrame(data,columns=['q'])
# print(tt)
# print(data)
# print(data2.iloc[:3,3])
# print(np.c_[data,data2.iloc[:3,3]])
cc = pd.concat([test,tt],axis=0)
print(cc)
