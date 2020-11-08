#%%
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import time
from fancyimpute import KNN
from sklearn.ensemble import IsolationForest
import warnings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')


#%%
# 缺失值分析
def missing_values(df):
    alldata_na = pd.DataFrame(df.isnull().sum(), columns={'missingNum'})
    alldata_na['existNum'] = len(df) - alldata_na['missingNum']
    alldata_na['sum'] = len(df)
    alldata_na['missingRatio'] = alldata_na['missingNum']/len(df)*100
    alldata_na['dtype'] = df.dtypes
    #ascending：默认True升序排列；False降序排列
    alldata_na = alldata_na[alldata_na['missingNum']>0].reset_index().sort_values(by=['missingNum','index'],ascending=[False,True])
    alldata_na.set_index('index',inplace=True)
    return alldata_na

# 按时间截取数据
def dataframe_cut(df, begin_time='', end_time=''):
    if begin_time != '':
        df = df.loc[df.TimeStample >= pd.to_datetime(begin_time, format='%Y-%m-%d %H:%M:%S')]
    if end_time != '':
        df = df.loc[df.TimeStample <= pd.to_datetime(end_time, format='%Y-%m-%d %H:%M:%S')] 
    return df.reset_index(drop=True)

#%%
# 读取csv
Qi_train=pd.read_csv('Qi_train4.csv')
Qi_test=pd.read_csv('Qi_test4.csv')
pre=pd.read_excel('lstm_sub.xlsx',index=None)



#%%
Qi_train['D_SUM']=np.sum((Qi_train['D1'],Qi_train['D2'],Qi_train['D3'],Qi_train['D4'],Qi_train['D5']),axis=0)
Qi_train['D_MEAN']=np.mean((Qi_train['D1'],Qi_train['D2'],Qi_train['D3'],Qi_train['D4'],Qi_train['D5']),axis=0)
Qi_train['D_STD']=np.std((Qi_train['D1'],Qi_train['D2'],Qi_train['D3'],Qi_train['D4'],Qi_train['D5']),axis=0)



#%%
# 格式化日期
Qi_train['TimeStample'] = pd.to_datetime(Qi_train['TimeStample'], format ='%Y-%m-%d %H:%M:%S')
Qi_test['TimeStample'] = pd.to_datetime(Qi_test['TimeStample'], format ='%Y-%m-%d %H:%M:%S')

#%%
# 获取训练集某段数据
y_start=2014
y_end=2017
Qi_train1=dataframe_cut(Qi_train, begin_time='{}-01-01 02:00:00'.format(y_start), end_time='{}-12-31 23:00:00'.format(y_end))

#%%
# 获取测试集某段数据
Qi_test18_01=dataframe_cut(Qi_test, begin_time='2018-01-01 02:00:00', end_time='2018-01-31 23:00:00')
Qi_test18_07=dataframe_cut(Qi_test, begin_time='2018-07-01 02:00:00', end_time='2018-07-31 23:00:00')
Qi_test18_10=dataframe_cut(Qi_test, begin_time='2018-10-01 02:00:00', end_time='2018-10-31 23:00:00')


#%%
# 合并训练集与测试集
Qi_all=pd.concat([Qi_train1,Qi_test18_01,Qi_test18_07], axis=0, ignore_index=True)

#%%
# 分割年月日时
Qi_all['year']=Qi_all['TimeStample'].dt.year
Qi_all['month']=Qi_all['TimeStample'].dt.month
Qi_all['day']=Qi_all['TimeStample'].dt.day
Qi_all['hour']=Qi_all['TimeStample'].dt.hour
Qi_all['week']=Qi_all['TimeStample'].dt.week


#%%
Qi_all['y_m'] = Qi_all.apply(
    lambda x: ' '.join(['{}_{}'.format(x['year'], x['month'])]), axis=1)


cols=['TimeStample', 'Qi', 'Rain_sum', 'Rain_mean', 'w', 'wd', 'year', 'month', 'week', 'day']
Qi_all1=Qi_all.loc[:,cols].copy().reset_index(drop=True)



#%%
# 分位数
def q10(x):
    return x.quantile(0.1)

def q20(x):
    return x.quantile(0.2)

def q25(x):
    return x.quantile(0.25)

def q30(x):
    return x.quantile(0.3)

def q40(x):
    return x.quantile(0.4)

def q60(x):
    return x.quantile(0.6)

def q70(x):
    return x.quantile(0.7)

def q75(x):
    return x.quantile(0.75)

def q80(x):
    return x.quantile(0.8)

def q90(x):
    return x.quantile(0.9)

# kurt
def kurt(col):
    return col.kurt()


group_train1=Qi_all1.loc[:,cols].copy().reset_index(drop=True)

#%%
# 周聚类统计特征
on='week'
stat_functions = ['min', 'max', 'mean', 'median', 'std', 'skew', kurt, q25, q75,]
stat_ways = ['min', 'max', 'mean', 'median', 'std', 'skew', 'kurt', 'q_25', 'q_75',]

stat_cols = ['Qi']
group_tmp_m = group_train1.groupby(on)[stat_cols].agg(stat_functions).reset_index()
group_tmp_m.columns = [on] + ['{}_{}_{}'.format(on,i, j) for i in stat_cols for j in stat_ways]
group_train1 = group_train1.merge(group_tmp_m, on=on, how='left')


#%%
# 构造数据集 滑窗y值
group_train1['y']=group_train1['Qi']
group_train1['y']=group_train1['y'].shift(-56)

#%%
# 数据集划分
drop_columns=["TimeStample",'y']
train_all = dataframe_cut(group_train1, begin_time='{}-01-01 02:00:00'.format('2014'), end_time='{}-12-31 23:00:00'.format('2017'))
valid_all = dataframe_cut(group_train1, begin_time='{}-01-01 02:00:00'.format('2014'), end_time='{}-12-31 23:00:00'.format('2014'))

features = train_all[:1].drop(drop_columns,axis=1).columns



#%%
x_train = train_all[features].values
x_valid = valid_all[features].values

y_train=train_all['y'].values
y_valid=valid_all['y'].values

trn_x=x_train
trn_y=y_train

val_x=x_valid
val_y=y_valid

train_matrix = lgb.Dataset(trn_x, label=trn_y)
valid_matrix = lgb.Dataset(val_x, label=val_y)



#%%
"""lgb模型训练"""
params = {
    'boosting_type': 'gbdt',
    'objective': 'mse',
    'metric':'mse',
    'num_leaves': 2**8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'learning_rate': 0.001,
    'max_depth': 3,
    'min_split_gain':0.7,
    'nthread': -1,
    'seed': 2020}

# 模型训练
model = lgb.train(params, train_matrix, 10000, 
valid_sets=[train_matrix, valid_matrix], verbose_eval=500,early_stopping_rounds=1e3)


# NSE评价指标
test_07 = dataframe_cut(group_train1, begin_time='{}-07-25 02:00:00'.format('2018'), end_time='{}-07-31 23:00:00'.format('2018'))
x_test1 = test_07[features].values
test_pred_08 = model.predict(x_test1, num_iteration=model.best_iteration).reshape(-1,)
test_y_08=pre.iloc[1,:].values
score1 = 1-0.65*(np.sum((test_pred_08[0:0+16]-test_y_08[0:0+16])**2)/np.sum((test_pred_08[0:0+16]-np.mean(test_y_08[0:0+16]))**2))-0.35*(np.sum((test_pred_08[0+16:0+56]-test_y_08[0+16:0+56])**2)/np.sum((test_pred_08[0+16:0+56]-np.mean(test_y_08[0+16:0+56]))**2))
print(score1)



features
pd.DataFrame({
        'column': features,
        'importance': model.feature_importance(),
    }).sort_values(by='importance')




#%%
# 预测
y_pre = model.predict(x_test, num_iteration=model.best_iteration).reshape(1,56)
zero = np.zeros((1,56))


#%%
y_pre[y_pre<0]=np.nan

y_pre=y_pre.reshape(56,1)

y_pre1=pd.DataFrame(y_pre,columns=['Qi'])

y_pre1['t']=0
for i in range(0,len(y_pre1)):
    y_pre1.loc[i,'t']=i

# 填充Qi流量缺失值
from fancyimpute import KNN
imp_cols=list(y_pre1.columns)
y_pre_df = pd.DataFrame(KNN(k=56).fit_transform(y_pre1.loc[:,imp_cols]),columns=imp_cols)

cols=['Qi']
y_pre1['Qi']=y_pre_df.loc[:,cols]

y_pre=y_pre.reshape(1,56)


#%%
# 将结果写入CSV文件
submit = pd.read_csv('./data/submission.csv',index_col=0)
submit.iloc[0] = zero[0]
submit.iloc[1] = y_pre[0]
submit.iloc[2] = zero[0]
#%%
submit.to_csv("sub_lgb25.csv")

