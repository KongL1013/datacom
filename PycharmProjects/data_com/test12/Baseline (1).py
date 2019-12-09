#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


user_info = pd.read_csv('member_info_0926.txt', header=None, sep='\t')
user_info.columns = ['用户id','性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问评率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值','关注话题','感兴趣话题']
for col in user_info.columns:
    print(col, len(user_info[col].unique()))
    
question_info = pd.read_csv('question_info_0926.txt', header=None, sep='\t')
question_info.columns = ['问题id','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']
for col in question_info.columns:
    print(col, len(question_info[col].unique()))


# 减内存

# In[3]:


# 减少内存占用
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype 
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            if str(col_type) == 'float64':
                df[col] = df[col].astype(np.float16)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))    
    return df


# In[4]:


pd.set_option('display.max_columns', None)


# 读数据

# In[5]:


import pandas as pd

# 导入数据
user_info = pd.read_csv('member_info_0926.txt', header=None, sep='\t')
question_info = pd.read_csv('question_info_0926.txt', header=None, sep='\t')
train = pd.read_csv('invite_info_0926.txt', header=None, sep='\t')
test = pd.read_csv('invite_info_evaluate_1_0926.txt', header=None, sep='\t')
answer_info=pd.read_csv('answer_info_0926.txt', header=None, sep='\t')
train.columns = ['qid', 'uid', 'dt', 'label']
test.columns = ['qid', 'uid', 'dt']
user_info.columns = ['uid', 'gender', 'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat', 'freq',                     'uf_b1', 'uf_b2','uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4',                     'uf_c5', 'score', 'follow_topic','inter_topic']
user_info  = user_info.drop(['creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat'], axis=1)

question_info.columns = ['qid', 'q_dt', 'title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']
answer_info.columns = ['aid', 'qid', 'uid', 'ans_dt', 'ans_t1', 'ans_t2', 'is_good', 'is_rec', 'is_dest', 'has_img',               'has_video', 'word_count', 'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',               'reci_xxx', 'reci_no_help', 'reci_dis']
answer_info=answer_info.drop(['is_dest'],axis=1)


# In[6]:


answer_info['a-day'] = answer_info['ans_dt'].apply(lambda x:int(x.split('-')[0].split('D')[1]))
answer_info['a-hour'] = answer_info['ans_dt'].apply(lambda x:int(x.split('-')[1].split('H')[1]))
train['i-day'] = train['dt'].apply(lambda x:int(x.split('-')[0].split('D')[1]))
train['i-hour'] = train['dt'].apply(lambda x:int(x.split('-')[1].split('H')[1]))
test['i-day'] = test['dt'].apply(lambda x:int(x.split('-')[0].split('D')[1]))
test['i-hour'] = test['dt'].apply(lambda x:int(x.split('-')[1].split('H')[1]))
question_info['q-day'] = question_info['q_dt'].apply(lambda x:int(x.split('-')[0].split('D')[1]))
question_info['q-hour'] = question_info['q_dt'].apply(lambda x:int(x.split('-')[1].split('H')[1]))
answer_info['a-time'] = answer_info.apply(lambda x:x['a-day']+x['a-hour']/24,axis=1)


# In[7]:


answer_info=answer_info.drop(['ans_dt'],axis=1)
train=train.drop(['dt'],axis=1)
test=test.drop(['dt'],axis=1)
question_info=question_info.drop(['q_dt'],axis=1)


# In[8]:


print(answer_info.columns)
print(question_info.columns)
print(user_info.columns)
print(train.columns)
print(test.columns)


# 用户特征增强：

# In[9]:


def yh1(x):
    if x=="-1":
        return []
    else:
        temp=x.split(',')
        return list(map(lambda y:int(y[1:]),temp))
def yh2(x):
    if x=="-1":
        return {}
    else:
        temp=x.split(',')
        temp1={}
        for i in temp:
            temp2=i.split(':')
            temp1[int(temp2[0][1:])]=float(temp2[1])
        return temp1
def yh3_mean(x):
    temp=list(x.values())
    if len(temp)==0:return 0
    else: return sum(temp)/len(temp)
def yh3_max(x):
    temp=list(x.values())
    if len(temp)==0:return 0
    else: return max(temp)
def yh3_min(x):
    temp=list(x.values())
    if len(temp)==0:return 0
    else: return min(temp)
def yh3_std(x):
    temp=np.array(list(x.values()))
    if len(temp)==0:return 0
    else: return np.std(temp)


# In[10]:


user_info['follow_topic']=user_info['follow_topic'].apply(yh1)
user_info['inter_topic']=user_info['inter_topic'].apply(yh2)


# In[11]:


user_info['len_follow_topic']=user_info['follow_topic'].apply(len)
user_info['len_inter_topic']=user_info['inter_topic'].apply(len)
user_info['mean_inter_topic']=user_info['inter_topic'].apply(yh3_mean)
user_info['max_inter_topic']=user_info['inter_topic'].apply(yh3_max)
user_info['min_inter_topic']=user_info['inter_topic'].apply(yh3_min)
user_info['std_inter_topic']=user_info['inter_topic'].apply(yh3_min)


# In[12]:


reduce_mem_usage(user_info)


# In[13]:


user_info.columns


# In[14]:


'''以上为用户基本统计特征，
1）性别：大量的未知特征
（2）访问频率：（可以当成连续特征，但是有一个未知项没有解决）
（3）二分类特征：A1，B1，C1，D1，E1
其中C1，E1特征数据分布较为异常（就是1的特别少）
（4）多分类特征：
（5）盐选值
大部分集中在200-400，小部分过大或者过小
以上是已有特征，主要考虑一下如何进行数据清洗之类的操作，13维
用户关注的话题数量
用户感兴趣的话题数量
用户兴趣度平均值、最小值、最大值

'''


# 问题特征增强

# In[15]:


def qs1(x):
    if x=="-1":
        return []
    else:
        temp=x.split(',')
        return list(map(lambda y:int(y[1:]),temp))
def qs2(x):
    if x=="-1":
        return []
    else:
        temp=x.split(',')
        return list(map(lambda y:int(y[2:]),temp))
question_info.head(3)


# In[16]:


question_info.head()


# In[17]:


question_info['topic']=question_info['topic'].apply(yh1)


# In[18]:


question_info['desc_t1']=question_info['desc_t1'].apply(qs2)
question_info['desc_t2']=question_info['desc_t2'].apply(qs1)
question_info['title_t1']=question_info['title_t1'].apply(qs2)
question_info['title_t2']=question_info['title_t2'].apply(qs1)


# In[19]:


question_info['len_qs_tit']=question_info['title_t1'].apply(len)
question_info['len_qs_des']=question_info['desc_t1'].apply(len)


# In[20]:


question_info['len_qs_topic']=question_info['topic'].apply(len)


# In[21]:


#question_info['qs_hdl']=question_info.apply(lambda x:0 if x['eva_qs_num']==0 else x['eva_qs_1_num']/x['eva_qs_num'],axis=1)
len(question_info.columns)
reduce_mem_usage(question_info)
111


# In[22]:


question_info.head(2)
'''
2.问题特征
问题相关话题数
创建时间:天数以及时间
问题描述长度:两种长度
embedding特征如何使用？？？
'''


# In[23]:


print(user_info.columns)
print(question_info.columns)


# answer特征

# In[24]:


answer_info = pd.merge(answer_info, question_info[['qid','q-day','topic']], on='qid')
answer_info = pd.merge(answer_info, user_info[['uid', 'gender', 'freq', 'uf_b1', 'uf_b2', 'uf_b3', 'uf_b4', 'uf_b5',       'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5','score', 'follow_topic',       'inter_topic']], on='uid')


# In[25]:


# 回答距提问的天数
answer_info['interval_qa'] = answer_info['a-day'] - answer_info['q-day']


# In[26]:


# 时间窗口划分
# train
# val
train_start = 3838
train_end = 3867

val_start = 3868
val_end = 3874

label_end = 3867
label_start = label_end - 6    # 3861-3867

#长期特征
train_label_feature_end_c = label_end - 7
train_label_feature_start_c = train_label_feature_end_c - 22    # 3838-3860
val_label_feature_end_c = val_start - 1
val_label_feature_start_c = val_label_feature_end_c - 22    # 3845-3867
#中期特征
train_label_feature_end_z = label_end - 7
train_label_feature_start_z = train_label_feature_end_z - 7    # 3854-3860
val_label_feature_end_z = val_start - 1
val_label_feature_start_z = val_label_feature_end_z - 7    # 3861-3867
#短期特征
train_label_feature_end_d = label_end - 7
train_label_feature_start_d = train_label_feature_end_d - 3    # 3857-3860
val_label_feature_end_d = val_start - 1
val_label_feature_start_d = val_label_feature_end_d - 3    # 3864-3867

#长期回答特征
train_ans_feature_end_c = label_end - 7
train_ans_feature_start_c = train_ans_feature_end_c - 50    # 3810-3860
val_ans_feature_end_c = val_start - 1
val_ans_feature_start_c = val_ans_feature_end_c - 50    # 3817-3867

#中期回答特征
train_ans_feature_end_z = label_end - 7
train_ans_feature_start_z = train_ans_feature_end_z - 20    # 3840-3860
val_ans_feature_end_z = val_start - 1
val_ans_feature_start_z = val_ans_feature_end_z - 20    # 3847-3867

#短期回答特征
train_ans_feature_end_d = label_end - 7
train_ans_feature_start_d = train_ans_feature_end_d - 7    # 3853-3860
val_ans_feature_end_d = val_start - 1
val_ans_feature_start_d = val_ans_feature_end_d - 7    # 3860-3867


# In[27]:


answer_info['a-day'].min()
# train 分布 3838-3867
# test 分布 3868-3874
# answer 范围 3807-3867
print(train_label_feature_end_c)
fea_cols = ['is_good', 'is_rec', 'has_img', 'has_video', 'word_count',            'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',            'reci_xxx', 'reci_no_help', 'reci_dis', 'interval_qa']
fea_cols1 = ['is_rec', 'word_count', 'reci_comment', 'reci_mark', 'reci_tks',            'reci_dis', 'interval_qa']


# #分布研究

# In[28]:


test_q_unique=test['qid'].unique()
test_u_unique=test['uid'].unique()
train_q_unique=train_label['qid'].unique()
train_u_unique=train_label['uid'].unique()
train_q_unique_a=train['qid'].unique()
train_u_unique_a=train['uid'].unique()
user_unique=user_info['uid'].unique()
quest_unique=question_info['qid'].unique()


# In[29]:


def jzcf_u(a,b):
    num=0
    dic_u={}
    for i in user_unique:
        dic_u[i]=[0,0]
    for i in a:
        dic_u[i][0]=1
    for i in b:
        dic_u[i][1]=1
    for i in dic_u:
        num+=dic_u[i][0]*dic_u[i][1]
    return len(a),len(b),num

def jzcf_q(a,b):
    num=0
    dic_q={}
    for i in quest_unique:
        dic_q[i]=[0,0]
    for i in a:
        dic_q[i][0]=1
    for i in b:
        dic_q[i][1]=1
    for i in dic_q:
        num+=dic_q[i][0]*dic_q[i][1]
    return len(a),len(b),num


# In[30]:


#train与test集作者数
#(904867, 613340, 467709)
#train与test集问题数
#(344665, 237167, 71742)
#所有训练集作者数
#(1358213, 613340, 552288)
#所有训练集问题数
#(926203, 237167, 79643)
#可见作者交叉程度较大，问题交叉程度较低
#jzcf_u(train_u_unique_a,test_u_unique)


# In[31]:


train_label = train[(train['i-day'] > train_label_feature_end_c)]
# 3860-3867  最后的train集


train_label_feature_c = train[(train['i-day'] >= train_label_feature_start_c) & (train['i-day'] <= train_label_feature_end_c)]
# 3838-3860
val_label_feature_c = train[(train['i-day'] >= val_label_feature_start_c) & (train['i-day'] <= val_label_feature_end_c)]
# 3845-3867
train_label_feature_z = train[(train['i-day'] >= train_label_feature_start_z) & (train['i-day'] <= train_label_feature_end_z)]
# 3853-3860
val_label_feature_z = train[(train['i-day'] >= val_label_feature_start_z) & (train['i-day'] <= val_label_feature_end_z)]
# 3860-3867
train_label_feature_d = train[(train['i-day'] >= train_label_feature_start_d) & (train['i-day'] <= train_label_feature_end_d)]
# 3857-3860
val_label_feature_d = train[(train['i-day'] >= val_label_feature_start_d) & (train['i-day'] <= val_label_feature_end_d)]
# 3864-3867

train_ans_feature_c = answer_info[(answer_info['a-day'] >= train_ans_feature_start_c) & (answer_info['a-day'] <= train_ans_feature_end_c)]
# 3810-3860
val_ans_feature_c = answer_info[(answer_info['a-day'] >= val_ans_feature_start_c) & (answer_info['a-day'] <= val_ans_feature_end_c)]
# 3817-3867
train_ans_feature_z = answer_info[(answer_info['a-day'] >= train_ans_feature_start_z) & (answer_info['a-day'] <= train_ans_feature_end_z)]
# 3840-3860
val_ans_feature_z = answer_info[(answer_info['a-day'] >= val_ans_feature_start_z) & (answer_info['a-day'] <= val_ans_feature_end_z)]
# 3847-3867
train_ans_feature_d = answer_info[(answer_info['a-day'] >= train_ans_feature_start_d) & (answer_info['a-day'] <= train_ans_feature_end_d)]
# 3840-3860
val_ans_feature_d = answer_info[(answer_info['a-day'] >= val_ans_feature_start_d) & (answer_info['a-day'] <= val_ans_feature_end_d)]
# 3847-3867


# In[ ]:





# In[32]:


def extract_feature1(target, label_feature, ans_feature,feature_col, text="c"):
    # 问题特征
    t1 = label_feature.groupby('qid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['qid', 'q_inv_mean'+'_'+text, 'q_inv_sum'+'_'+text, 'q_inv_std'+'_'+text, 'q_inv_count'+'_'+text]
    target = pd.merge(target, t1, on='qid', how='left')

    # 用户特征
    t1 = label_feature.groupby('uid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['uid', 'u_inv_mean'+'_'+text, 'u_inv_sum'+'_'+text, 'u_inv_std'+'_'+text, 'u_inv_count'+'_'+text]
    target = pd.merge(target, t1, on='uid', how='left')
    
    
    # train_size = len(train)
    # data = pd.concat((train, test), sort=True)

    # 回答部分特征

    t1 = ans_feature.groupby('qid')['aid'].count().reset_index()
    t1.columns = ['qid', 'q_ans_count'+'_'+text]
    target = pd.merge(target, t1, on='qid', how='left')

    t1 = ans_feature.groupby('uid')['aid'].count().reset_index()
    t1.columns = ['uid', 'u_ans_count'+'_'+text]
    target = pd.merge(target, t1, on='uid', how='left')

    for col in feature_col:
        t1 = ans_feature.groupby('uid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['uid', f'u_{col}_sum'+'_'+text, f'u_{col}_max'+'_'+text, f'u_{col}_mean'+'_'+text]
        target = pd.merge(target, t1, on='uid', how='left')

        t1 = ans_feature.groupby('qid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['qid', f'q_{col}_sum'+'_'+text, f'q_{col}_max'+'_'+text, f'q_{col}_mean'+'_'+text]
        target = pd.merge(target, t1, on='qid', how='left')
        print("extract %s", col)
    return target


# In[30]:


train_label = extract_feature1(train_label, train_label_feature_c, train_ans_feature_c,feature_col=fea_cols1,text="c")
test = extract_feature1(test, val_label_feature_c, val_ans_feature_c,feature_col=fea_cols1,text="c")
#长期特征
train_label = extract_feature1(train_label, train_label_feature_z, train_ans_feature_z,feature_col=fea_cols1,text="z")
test = extract_feature1(test, val_label_feature_z, val_ans_feature_z,feature_col=fea_cols1,text="z")
#中期特征
train_label = extract_feature1(train_label, train_label_feature_d, train_ans_feature_d,feature_col=fea_cols1,text="d")
test = extract_feature1(test, val_label_feature_d, val_ans_feature_d,feature_col=fea_cols1,text="d")
#短期窗口


# In[31]:


reduce_mem_usage(test)
reduce_mem_usage(train_label)


# In[32]:


# merge user
train_label = pd.merge(train_label, user_info, on='uid', how='left')
test = pd.merge(test, user_info, on='uid', how='left')
#logging.info("train shape %s, test shape %s", train_label.shape, test.shape)


# In[33]:


data = pd.concat((train_label, test), axis=0, sort=True)
data=pd.merge(data,question_info,on="qid",how='left')
del train_label, test


# In[34]:


from sklearn.preprocessing import LabelEncoder
class_feat =  ['uid','qid','gender', 'freq','uf_c1','uf_c2','uf_c3','uf_c4','uf_c5']
encoder = LabelEncoder()
for feat in class_feat:
    encoder.fit(data[feat])
    data[feat] = encoder.transform(data[feat])


# In[35]:


reduce_mem_usage(data)


# In[36]:


for feat in ['gender', 'freq','uf_c1','uf_c2','uf_c3','uf_c4','uf_c5']:
    col_name = '{}_count'.format(feat)
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data.loc[data[col_name] < 2, feat] = -1
    data[feat] += 1
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())


# In[37]:


reduce_mem_usage(data)


# In[38]:


for feat in ['uid','qid','uf_b1','uf_b2','uf_b3','uf_b4','uf_b5']:
    col_name = '{}_count'.format(feat)
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data.loc[data[col_name] < 2, feat] = -1
    data[feat] += 1
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())


# In[39]:


len(data.columns)


# In[64]:


import pickle
path_topic = r"./data/topic.pkl"
with open(path_topic,'rb') as file:
    data_topic=pickle.load(file).values


# In[40]:


import time
t_start = time.time()
data["interval_qi"] = data.apply(lambda x:x['i-day'] - x['q-day'], axis=1)
print(-t_start + time.time())


# In[41]:


import pickle
with open('./train11.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[ ]:




