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


data_use = data[:2593669]
data_evaluate = data[2593669:]


isnull_index = data_use['title_ans_t1_max'].isnull()
# print(isnull_index)
data_isnull = data_use[isnull_index]
data_notnull = data_use[~isnull_index]

# print(len(data_notnull['title_ans_t1_max']))
print('len(data_notnull.index) = ',len(data_notnull.index))
print('len(data_isnull.index) = ',len(data_isnull.index))


evaluate_isnull_index = data_evaluate['title_ans_t1_max'].isnull()
evaluate_isnull = data_evaluate[evaluate_isnull_index]
evaluate_notnull = data_evaluate[~evaluate_isnull_index]

print("evaluate_isnull = ",len(evaluate_isnull))
print("evaluate_notnull = ",len(evaluate_notnull))


data_isnull = data_isnull.reset_index()
data_notnull = data_notnull.reset_index()

evaluate_isnull = evaluate_isnull.reset_index()
evaluate_notnull = evaluate_notnull.reset_index()


X_isnull = data_isnull.drop(['label','index','title_ans_t1_max',	'title_ans_t1_min',	'title_ans_t1_sum',	'title_ans_t1_mean'	,'title_ans_t1_std'	,'title_title_t1_max',	'title_title_t1_min',	'title_title_t1_sum' ,'title_title_t1_mean',	'title_title_t1_std',	'title_desc_t1_max'	,'title_desc_t1_min',	'title_desc_t1_sum',	'title_desc_t1_mean',	'title_desc_t1_std',	'desc_ans_t1_max'	,'desc_ans_t1_min',	'desc_ans_t1_sum'	,'desc_ans_t1_mean'	,'desc_ans_t1_std',	'desc_title_t1_max',	'desc_title_t1_min'	,'desc_title_t1_sum'	,'desc_title_t1_mean'	,'desc_title_t1_std',	'desc_desc_t1_max'	,'desc_desc_t1_min',	'desc_desc_t1_sum',	'desc_desc_t1_mean',	'desc_desc_t1_std'	,'title_ans_t2_max'	,'title_ans_t2_min'	,'title_ans_t2_sum'	,'title_ans_t2_mean'	,'title_ans_t2_std',	'title_title_t2_max',	'title_title_t2_min',	'title_title_t2_sum',	'title_title_t2_mean',	'title_title_t2_std',	'title_desc_t2_max',	'title_desc_t2_min'	,'title_desc_t2_sum'	,'title_desc_t2_mean',	'title_desc_t2_std',	'desc_ans_t2_max',	'desc_ans_t2_min',	'desc_ans_t2_sum',	'desc_ans_t2_mean',	'desc_ans_t2_std',	'desc_title_t2_max'	,'desc_title_t2_min'	,'desc_title_t2_sum',	'desc_title_t2_mean'	,'desc_title_t2_std',	'desc_desc_t2_max',	'desc_desc_t2_min'	,'desc_desc_t2_sum'	,'desc_desc_t2_mean',	'desc_desc_t2_std'], axis=1).values
y_isnull = data_isnull['label'].values

X_notnull = data_notnull.drop(['label','index'], axis=1).values
y_notnull = data_notnull['label'].values

evaluate_X_isnull = evaluate_isnull.drop(['label','index','title_ans_t1_max',	'title_ans_t1_min',	'title_ans_t1_sum',	'title_ans_t1_mean'	,'title_ans_t1_std'	,'title_title_t1_max',	'title_title_t1_min',	'title_title_t1_sum' ,'title_title_t1_mean',	'title_title_t1_std',	'title_desc_t1_max'	,'title_desc_t1_min',	'title_desc_t1_sum',	'title_desc_t1_mean',	'title_desc_t1_std',	'desc_ans_t1_max'	,'desc_ans_t1_min',	'desc_ans_t1_sum'	,'desc_ans_t1_mean'	,'desc_ans_t1_std',	'desc_title_t1_max',	'desc_title_t1_min'	,'desc_title_t1_sum'	,'desc_title_t1_mean'	,'desc_title_t1_std',	'desc_desc_t1_max'	,'desc_desc_t1_min',	'desc_desc_t1_sum',	'desc_desc_t1_mean',	'desc_desc_t1_std'	,'title_ans_t2_max'	,'title_ans_t2_min'	,'title_ans_t2_sum'	,'title_ans_t2_mean'	,'title_ans_t2_std',	'title_title_t2_max',	'title_title_t2_min',	'title_title_t2_sum',	'title_title_t2_mean',	'title_title_t2_std',	'title_desc_t2_max',	'title_desc_t2_min'	,'title_desc_t2_sum'	,'title_desc_t2_mean',	'title_desc_t2_std',	'desc_ans_t2_max',	'desc_ans_t2_min',	'desc_ans_t2_sum',	'desc_ans_t2_mean',	'desc_ans_t2_std',	'desc_title_t2_max'	,'desc_title_t2_min'	,'desc_title_t2_sum',	'desc_title_t2_mean'	,'desc_title_t2_std',	'desc_desc_t2_max',	'desc_desc_t2_min'	,'desc_desc_t2_sum'	,'desc_desc_t2_mean',	'desc_desc_t2_std'], axis=1).values
evaluate_X_notnull = evaluate_notnull.drop(['label','index'], axis=1).values
#train


'''
X_train, X_test, y_train, y_test = train_test_split(X_isnull, y_isnull, test_size=0.1)


print("start isnull_model_xgboost")
isnull_model_xgboost = XGBClassifier(
                      max_depth=depth,
                      learning_rate=0.01,
                      n_estimators=steps,
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
isnull_model_xgboost.fit(X_train,y_train,
              eval_metric='auc',
              eval_set=[(X_train, y_train),(X_test, y_test)],#, (X_test, y_test)
              #categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
              early_stopping_rounds=200)
# save model to file
pickle.dump(isnull_model_xgboost, open("./model/mydata1_isnull_xgboost.pickle.dat", "wb"))


# model_xgboost = pickle.load(open("./model/model_xgboost.pickle.dat", "rb"))


y_pred_test = isnull_model_xgboost.predict(X_test)
predictions = [round(value) for value in y_pred_test]
accuracy = accuracy_score(y_test, predictions)
print("FINALL TEST Accuracy: %.2f%%" % (accuracy * 100.0))

X_train, X_test, y_train, y_test = train_test_split(X_notnull, y_notnull, test_size=0.1)
print("start notnull_model_xgboost")
notnull_model_xgboost = XGBClassifier(
                      max_depth=depth,
                      learning_rate=0.01,
                      n_estimators=steps,
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
notnull_model_xgboost.fit(X_train,y_train,
              eval_metric='auc',
              eval_set=[(X_train, y_train),(X_test, y_test)],#, (X_test, y_test)
              #categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
              early_stopping_rounds=200)
# save model to file
pickle.dump(notnull_model_xgboost, open("./model/mydata1_notnull_xgboost.pickle.dat", "wb"))


# model_xgboost = pickle.load(open("./model/model_xgboost.pickle.dat", "rb"))


y_pred_test = notnull_model_xgboost.predict(X_test)
predictions = [round(value) for value in y_pred_test]
accuracy = accuracy_score(y_test, predictions)
print("FINALL TEST Accuracy: %.2f%%" % (accuracy * 100.0))

'''




from sklearn.model_selection import StratifiedKFold

n_fold = 5
steps = 2200
depth = 10


skf = StratifiedKFold(n_splits=n_fold, random_state=2020, shuffle=False)

# isnul 5 fold
# X_evaluate = data[2593669:].drop(['label'], axis=1).values
isnull_pred = np.zeros([evaluate_X_isnull.shape[0],2])
cnt = 1

for index ,(train_index,test_index) in enumerate(skf.split(X_isnull , y_isnull)): #训练数据五折
    print("start train time = {cnt}".format(cnt=cnt))
    print("train_index = ",train_index)
    print("train_index_% = ", len(train_index)/len(X_isnull))
    train_x, test_x, train_y, test_y = X_isnull[train_index], X_isnull[test_index], y_isnull[
        train_index], y_isnull[test_index]
    print("start training")
    # if cnt%2 != 0:
    if True:
        print("model_xgboost")
        model = XGBClassifier(
            max_depth=9,
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
                          early_stopping_rounds=200)
        print("save {} model!!!".format(cnt))
        pickle.dump(model, open("./model/isnull_mydata_model_xgboost{}.pickle.dat".format(cnt), "wb"))
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

    isnull_pred += model.predict_proba(evaluate_X_isnull)
    print('len X_evaluate =',len(evaluate_X_isnull))
    print("len y_pred=",len(isnull_pred))
    print("y_pred")
    print(isnull_pred[:5, :]/cnt)
    cnt += 1

isnull_pred = (isnull_pred/n_fold)[:,1]


### not null
skf = StratifiedKFold(n_splits=n_fold, random_state=2020, shuffle=False)
notnull_pred = np.zeros([evaluate_X_notnull.shape[0],2])
cnt = 1

for index ,(train_index,test_index) in enumerate(skf.split(X_notnull , y_notnull)): #训练数据五折
    print("start train time = {cnt}".format(cnt=cnt))
    print("train_index = ",train_index)
    print("train_index_% = ", len(train_index)/len(X_notnull))
    train_x, test_x, train_y, test_y = X_notnull[train_index], X_notnull[test_index], y_notnull[
        train_index], y_notnull[test_index]
    print("start training")
    # if cnt%2 != 0:
    if True:
        print("model_xgboost")
        model = XGBClassifier(
            max_depth=depth,
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
                          early_stopping_rounds=200)
        print("save {} model!!!".format(cnt))
        pickle.dump(model, open("./model/notnull_mydata_model_xgboost{}.pickle.dat".format(cnt), "wb"))
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

    notnull_pred += model.predict_proba(evaluate_X_notnull)
    print('len X_evaluate =',len(evaluate_X_notnull))
    print("len y_pred=",len(notnull_pred))
    print("y_pred")
    print(notnull_pred[:5, :]/cnt)
    cnt += 1

notnull_pred = (notnull_pred/n_fold)[:,1]



#save data


# isnull_pred = isnull_model_xgboost.predict_proba(evaluate_X_isnull)[:,1]
# print("isnull_pred\n",isnull_pred)
# notnull_pred = notnull_model_xgboost.predict_proba(evaluate_X_notnull)[:,1]
# print("notnull_pred\n",notnull_pred)

store_isnull = pd.DataFrame()
store_notnull = pd.DataFrame()
store_isnull['index'] = evaluate_isnull['index']
store_isnull['y_pred'] = isnull_pred
print(store_isnull.head())

store_notnull['index'] = evaluate_notnull['index']
store_notnull['y_pred'] = notnull_pred
print(store_notnull.head())


store = pd.concat([store_isnull,store_notnull],axis=0)
print('before sort',store.head(20))
store = store.sort_values(by="index" , ascending=True)
print('after sort',store.head(40))

y_pred = store['y_pred'].values

########store data
with open('test1.pkl', 'rb') as file:
    result_append = pickle.load(file)

# print(data[2593669:].head())

result_append['Score'] = y_pred

print(result_append.head())

result_append.to_csv('./model/mydata2.txt', header=False, index=False, sep='\t')

