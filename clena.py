#%%
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import time
from sklearn.ensemble import IsolationForest
import warnings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')


# 按时间截取数据
def dataframe_cut(df, begin_time='', end_time=''):
    if begin_time != '':
        df = df.loc[df.TimeStample >= pd.to_datetime(begin_time, format='%Y-%m-%d %H:%M:%S')]
    if end_time != '':
        df = df.loc[df.TimeStample <= pd.to_datetime(end_time, format='%Y-%m-%d %H:%M:%S')] 
    return df.reset_index(drop=True)


#%%
# 入库流量数据
# 1个总入库流量监测点  每3个小时
data = pd.read_excel('./data/入库流量数据.xlsx')
data['TimeStample'] = pd.to_datetime(data['TimeStample'], format ='%Y-%m-%d %H:%M:%S')
# 获取训练集某段数据
y_start=2013
y_end=2017
Qi_train1=dataframe_cut(data, begin_time='{}-01-01 02:00:00'.format(y_start), end_time='{}-12-31 23:00:00'.format(y_end))

Qi_test1 = pd.read_excel('./final_data/入库流量数据.xlsx')
Qi_test1['TimeStample'] = pd.to_datetime(Qi_test1['TimeStample'], format ='%Y-%m-%d %H:%M:%S')




# %%
# 遥测站降雨数据
# 39个雨量观测站  每1个小时
Raindata = pd.read_excel('./data/遥测站降雨数据.xlsx')
raindata_new = pd.DataFrame()
raindata_new['TimeStample'] = Raindata['TimeStample'].copy()
Raindata.drop('TimeStample',axis=1,inplace=True)
# 将所有测点的数据求和作为总降雨
raindata_new['Rain_sum'] = Raindata.apply(lambda x: x.sum(), axis=1)
Rain_train=dataframe_cut(raindata_new, begin_time='{}-01-01 00:00:00'.format(y_start), end_time='{}-12-31 23:00:00'.format(y_end))

#%%
Raindata_test = pd.read_excel('./final_data/遥测站降雨数据.xlsx')
raindata_new_test = pd.DataFrame()
raindata_new_test['TimeStample'] = Raindata_test['TimeStample'].copy()
Raindata_test.drop('TimeStample',axis=1,inplace=True)
# 将所有测点的数据求和作为总降雨
raindata_new_test['Rain_sum'] = Raindata_test.apply(lambda x: x.sum(), axis=1)
Rain_test1 = raindata_new_test.copy().reset_index(drop=True)



#%%
# 获取Qi完整索引
index = raindata_new[2:len(Rain_train):3][['TimeStample']].copy().reset_index(drop=True)
Qi_train1=index.merge(Qi_train1,on='TimeStample',how='left')

# 构造时间t
temp=Qi_train1.copy()
t=[]
for i in range(2,len(Rain_train),3):
    t.append(i)
temp['t']=t
#%%
# 填充Qi流量缺失值
from fancyimpute import KNN
imp_cols=['t', 'Qi']
Qi_train1_df = pd.DataFrame(KNN(k=8).fit_transform(temp[imp_cols]), columns=imp_cols)

imp_cols=['Qi']
Qi_train1.loc[:,imp_cols]=Qi_train1_df.loc[:,imp_cols]



#%%
# 遥测站数据合并
# 遥测站数据重采样
Rain_train.set_index('TimeStample', inplace=True)
# 流量的采样频率为（每3个小时），而降雨为每1个小时
Rain_train = Rain_train.resample('3H').sum()
Qi_train1['Rain_sum'] = Rain_train['Rain_sum'].values

Rain_test1.set_index('TimeStample', inplace=True)
a1 = Rain_test1[0:31*24].resample('3H').sum()
a2 = Rain_test1[31*24:2*31*24].resample('3H').sum()
a3 = Rain_test1[2*31*24:3*31*24].resample('3H').sum()
a4 = Rain_test1[3*31*24:4*31*24].resample('3H').sum()
a5 = Rain_test1[4*31*24:5*31*24].resample('3H').sum()
Qi_test1['Rain_sum'] = pd.concat([a1,a2,a3,a4,a5],axis=0,ignore_index=True)



#%%
# 环境表数据
# T气温、w风向、wd风速  每天
Environmentdata = pd.read_excel('./data/环境表.xlsx')
# 填充缺失
Environmentdata['T'].fillna(method='ffill',inplace=True)
Environmentdata['w'].fillna(method='ffill',inplace=True)
# wd归一化
from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler()
Environmentdata['wd'] = ss.fit_transform(Environmentdata['wd'].values.reshape(-1,1))
Environmentdata['TimeStample'] = pd.to_datetime(Environmentdata['TimeStample'], format ='%Y-%m-%d')
Environmentdata=dataframe_cut(Environmentdata, begin_time='{}-01-01'.format(y_start), end_time='{}-12-31'.format(y_end))

#%%
# 测试数据
Environmentdata_test = pd.read_excel('./final_data/环境表.xlsx')
# 填充缺失
Environmentdata_test['T'].fillna(method='ffill',inplace=True)
Environmentdata_test['w'].fillna(method='ffill',inplace=True)
# wd归一化
from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler()
Environmentdata_test['wd'] = ss.fit_transform(Environmentdata_test['wd'].values.reshape(-1,1))


#%%
# 环境表数据重采样
Environment = pd.DataFrame()
Environment['T'] = np.zeros(len(Environmentdata)*8)
Environment['w'] = np.zeros(len(Environmentdata)*8)
Environment['wd'] = np.zeros(len(Environmentdata)*8)
for i in range(len(Environment)):
    Environment['T'][i] = Environmentdata['T'][int(i/8)]
    Environment['w'][i] = Environmentdata['w'][int(i/8)]
    Environment['wd'][i] = Environmentdata['wd'][int(i/8)]

#%%
Environment_test = pd.DataFrame()
Environment_test['T'] = np.zeros(len(Environmentdata_test)*8)
Environment_test['w'] = np.zeros(len(Environmentdata_test)*8)
Environment_test['wd'] = np.zeros(len(Environmentdata_test)*8)
for i in range(len(Environment_test)):
    Environment_test['T'][i] = Environmentdata_test['T'][int(i/8)]
    Environment_test['w'][i] = Environmentdata_test['w'][int(i/8)]
    Environment_test['wd'][i] = Environmentdata_test['wd'][int(i/8)]


#%%
# 环境数据合并
Qi_train1['T'] = Environment['T'][0:len(Qi_train1)].values
Qi_train1['w'] = Environment['w'][0:len(Qi_train1)].values
Qi_train1['wd'] = Environment['wd'][0:len(Qi_train1)].values

#%%
Qi_test1['T']= Environment_test['T'][0:len(Qi_test1)].values
Qi_test1['w']= Environment_test['w'][0:len(Qi_test1)].values
Qi_test1['wd']= Environment_test['wd'][0:len(Qi_test1)].values



#%%
# 降雨预报数据
# 未来5日  每天
Weatherdata = pd.read_excel('./data/降雨预报数据.xlsx')

# 获取完整索引
index2 = Environmentdata[['TimeStample']].copy().reset_index(drop=True)
index2['TimeStample'] = pd.to_datetime(index2['TimeStample'], format ='%Y-%m-%d')
Weatherdata=index2.merge(Weatherdata,on='TimeStample',how='left')

# 构造时间t
temp2=Weatherdata.copy()
t=[]
for i in range(0,len(Weatherdata)):
    t.append(i)
temp2['t']=t

# 填充未来5天降雨缺失值
from fancyimpute import KNN
imp_cols=['t', 'D1', 'D2', 'D3', 'D4', 'D5']
Weatherdata_df = pd.DataFrame(KNN(k=8).fit_transform(temp2[imp_cols]), columns=imp_cols)

imp_cols=['D1', 'D2', 'D3', 'D4', 'D5']
Weatherdata.loc[:,imp_cols]=Weatherdata_df.loc[:,imp_cols]

#%%
# 测试数据
Weatherdata_test = pd.read_excel('./final_data/降雨预报数据.xlsx')

# 获取完整索引
index2 = Environmentdata_test[['TimeStample']].copy().reset_index(drop=True)
index2['TimeStample'] = pd.to_datetime(index2['TimeStample'], format ='%Y-%m-%d')
Weatherdata_test=index2.merge(Weatherdata_test,on='TimeStample',how='left')

# 构造时间t
temp2_test=Weatherdata_test.copy()
t=[]
for i in range(0,len(Weatherdata_test)):
    t.append(i)
temp2_test['t']=t

# 填充未来5天降雨缺失值
from fancyimpute import KNN
imp_cols=['t', 'D1', 'D2', 'D3', 'D4', 'D5']
Weatherdata_df_test = pd.DataFrame(KNN(k=8).fit_transform(temp2_test[imp_cols]), columns=imp_cols)

imp_cols=['D1', 'D2', 'D3', 'D4', 'D5']
Weatherdata_test.loc[:,imp_cols]=Weatherdata_df_test.loc[:,imp_cols]


#%%
# 降雨预报数据重采样
Weather = pd.DataFrame()
Weather['D1'] = np.zeros(len(Weatherdata)*8)
Weather['D2'] = np.zeros(len(Weatherdata)*8)
Weather['D3'] = np.zeros(len(Weatherdata)*8)
Weather['D4'] = np.zeros(len(Weatherdata)*8)
Weather['D5'] = np.zeros(len(Weatherdata)*8)
for i in range(len(Weather)):
    Weather['D1'][i] = Weatherdata['D1'][int(i/8)]
    Weather['D2'][i] = Weatherdata['D2'][int(i/8)]
    Weather['D3'][i] = Weatherdata['D3'][int(i/8)]
    Weather['D4'][i] = Weatherdata['D4'][int(i/8)]
    Weather['D5'][i] = Weatherdata['D5'][int(i/8)]

#%%
# 降雨预报数据重采样
Weather_test = pd.DataFrame()
Weather_test['D1'] = np.zeros(len(Weatherdata_test)*8)
Weather_test['D2'] = np.zeros(len(Weatherdata_test)*8)
Weather_test['D3'] = np.zeros(len(Weatherdata_test)*8)
Weather_test['D4'] = np.zeros(len(Weatherdata_test)*8)
Weather_test['D5'] = np.zeros(len(Weatherdata_test)*8)
for i in range(len(Weather_test)):
    Weather_test['D1'][i] = Weatherdata_test['D1'][int(i/8)]
    Weather_test['D2'][i] = Weatherdata_test['D2'][int(i/8)]
    Weather_test['D3'][i] = Weatherdata_test['D3'][int(i/8)]
    Weather_test['D4'][i] = Weatherdata_test['D4'][int(i/8)]
    Weather_test['D5'][i] = Weatherdata_test['D5'][int(i/8)]


#%%
# 降雨预报数据合并
Qi_train1['D1'] = Weather['D1'][0:len(Qi_train1)].values
Qi_train1['D2'] = Weather['D2'][0:len(Qi_train1)].values
Qi_train1['D3'] = Weather['D3'][0:len(Qi_train1)].values
Qi_train1['D4'] = Weather['D4'][0:len(Qi_train1)].values
Qi_train1['D5'] = Weather['D5'][0:len(Qi_train1)].values

#%%
Qi_test1['D1']= Weather['D1'][0:len(Qi_test1)].values
Qi_test1['D2']= Weather['D2'][0:len(Qi_test1)].values
Qi_test1['D3']= Weather['D3'][0:len(Qi_test1)].values
Qi_test1['D4']= Weather['D4'][0:len(Qi_test1)].values
Qi_test1['D5']= Weather['D5'][0:len(Qi_test1)].values



#%%
# 输出csv
Qi_train1.to_csv('Qi_train_final1.csv',index=None)
Qi_test1.to_csv('Qi_test_final1.csv',index=None)




