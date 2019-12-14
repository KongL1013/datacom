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

X_evaluate = data[2593669:].drop(['label'], axis=1).values
y_pred = np.zeros([X_evaluate.shape[0],2])


for i in range():
    loaded_model1 = pickle.load(open("all_xgboost{}.pickle.dat ".format(i), "rb"))
    total_xg_pred1 = loaded_model1.predict_proba(X_evaluate)

