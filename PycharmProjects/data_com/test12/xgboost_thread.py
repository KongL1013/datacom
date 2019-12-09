#!/usr/bin/env python
# coding: utf-8

def do_xgboost():
    from xgboost import XGBClassifier
    global model_xgboost
    import lightgbm as lgb
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import seaborn as sns

    pd.set_option('display.max_columns', None)

    with open('train11.pkl', 'rb') as file:
        data = pickle.load(file)

    def fill_null(data, col, null_value):
        use_avg = data[data[col] != null_value][col].mean()
        data.loc[data[col] == null_value, col] = use_avg
        return data

    def fill_null_nan(data, col, null_value):
        use_avg = np.nan
        data.loc[data[col] == null_value, col] = use_avg
        return data

    with open('train10.pkl', 'rb') as file:
        data1 = pickle.load(file)

    print(len(data1), len(data))

    data['topic_sim0_max'] = data1['topic_sim0'].apply(lambda x: x[0])
    data['topic_sim0_avg'] = data1['topic_sim0'].apply(lambda x: x[1])
    data['topic_sim0_min'] = data1['topic_sim0'].apply(lambda x: x[2])
    data['topic_sim0_std'] = data1['topic_sim0'].apply(lambda x: x[3])
    data['topic_sim0_num'] = data1['topic_sim0'].apply(lambda x: x[4])

    data['topic_sim1_max'] = data1['topic_sim1'].apply(lambda x: x[0])
    data['topic_sim1_avg'] = data1['topic_sim1'].apply(lambda x: x[1])
    data['topic_sim1_min'] = data1['topic_sim1'].apply(lambda x: x[2])
    data['topic_sim1_std'] = data1['topic_sim1'].apply(lambda x: x[3])
    data['topic_sim1_num'] = data1['topic_sim1'].apply(lambda x: x[4])
    data['topic_sim1_max1'] = data1['topic_sim1'].apply(lambda x: x[5])
    data['topic_sim1_min1'] = data1['topic_sim1'].apply(lambda x: x[6])

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

    data = data.drop(['follow_topic', 'inter_topic', 'topic', 'title_t1', 'title_t2', 'desc_t1', 'desc_t2'], axis=1)

    print(len(data) - 1141683)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    # 划分训练集和测试集
    # y_train = data[:train.shape[0]]['label'].values
    # X_train = data[:train.shape[0]].drop(['label'], axis=1).values

    # X_test = data[train.shape[0]:].drop(['label'], axis=1).values
    X = data[:2593669].drop(['label'], axis=1).values
    y = data[:2593669]['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    print("start xgboost")
    model_xgboost = XGBClassifier(
        max_depth=13,
        learning_rate=0.01,
        n_estimators=2000,
        min_child_weight=5,
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
    try:
        model_xgboost.fit(X_train, y_train,
                          eval_metric='auc',
                          eval_set=[(X_train, y_train), (X_test, y_test)],  # , (X_test, y_test)
                          # categorical_feature=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29],
                          early_stopping_rounds=50)
        # save model to file
        pickle.dump(model_xgboost, open("model_xgboost.pickle.dat", "wb"))
    except KeyboardInterrupt:
        print('START TO SAVE RESULT!!!!!!!!!!!!!')
        # load model from file
        # loaded_model = pickle.load(open("model_xgboost.pickle.dat", "rb"))
        y_pred_test = model_xgboost.predict(X_test)
        predictions = [round(value) for value in y_pred_test]
        accuracy = roc_auc_score(y_test, predictions)
        print("FINALL Accuracy: %.2f%%" % (accuracy * 100.0))

        X_evaluate = data[2593669:].drop(['label'], axis=1).values
        y_pred = model_xgboost.predict_proba(X_evaluate)
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
        raise
if __name__ == '__main__':
    do_xgboost()



