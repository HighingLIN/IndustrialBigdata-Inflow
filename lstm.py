#%%
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import time
import warnings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')

#%%
# 导入keras
import tensorflow as tf
import random as rn
#%%
def seed_tensorflow(seed=1):
    rn.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(1)

seed_tensorflow(1)
#%%
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM,Dense,TimeDistributed


#%%
# GPU 显存自动分配
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction=0.8
tf.compat.v1.Session(config=config)



#%%
# 读取csv
Qi_train1=pd.read_csv('Qi_train_final1.csv')
Qi_test1=pd.read_csv('Qi_test_final1.csv')



#%%
# 构造时间监督学习的训练集和预测集
# step步长为1，n_input为序列长度，n_out为输出的行数，n_features为要使用的特征数
def to_supervised(train, n_input, n_out, n_features):
    data = train.copy()
    X, y = list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end, 0:n_features])  # 使用几个特征
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return np.array(X), np.array(y)

def to_supervised_test(train, n_input, n_out, n_features):
    data = train.copy()
    X = list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        if in_end <= len(data):
            X.append(data[in_start:in_end, 0:n_features])  # 使用几个特征
        in_start += 1
    return np.array(X)

def to_supervised_valid(train, n_input, n_out, n_features):
    data = train.copy()
    X, y = list(), list()
    in_start = 0
    out_start = 248
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = out_start + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end, 0:n_features])  # 使用几个特征
            y.append(data[out_start:out_end, 0])
        in_start += 1
        out_start += 1
    return np.array(X), np.array(y)

#%%
features=['Qi', 'Rain_sum', 'w', 'wd']

# 复制Qi_train1副本
trainData1=Qi_train1.copy().reset_index(drop=True)
# 获取数值
trainData1=trainData1.loc[:,features].values

# 划分训练集与验证集
train=trainData1[2920:].copy()
valid=trainData1[2920:2920*2].copy()

n_weeks = 4   # 可调
n_input = n_weeks*7*8   #4*7*8=224
n_out = 7*8
n_features = len(features)

train_x, train_y = to_supervised(train, n_input, n_out,n_features)
valid_x, valid_y = to_supervised(valid, n_input, n_out, n_features)



#%%
# 构造模型 初始化
np.random.seed(1)
rn.seed(1)
tf.random.set_seed(1)

# 构造模型
model = Sequential()

drop=0.

model.add(keras.layers.Bidirectional(LSTM(32, dropout=drop, return_sequences=True), input_shape=(n_input, n_features),))
model.add(keras.layers.Bidirectional(LSTM(16, dropout=drop),))

model.add(Dense(n_out))
model.summary()

#%%
# 编译参数
#打印记录
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print(epoch, "\t", logs)

opt = keras.optimizers.Adam(learning_rate=0.1,)
model.compile(optimizer=opt, loss='mse', metrics=['mae'])

#早停函数
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0, mode='min')

# 动态学习率
reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,  min_delta=1e-4, verbose=0, mode='min')

#%%
# 模型训练
start=time.time()
model.fit(train_x, train_y, epochs=200, batch_size=128, shuffle=False, 
              validation_data = (valid_x, valid_y), 
              verbose=1, callbacks=[PrintDot(), reduce_lr, early_stop])
end=time.time()
print(end-start)

#%%
# NSE评价指标
ypre = model.predict(valid_x).reshape((2641,56))

score1 = 1-0.65*(np.sum((valid_y[24,:16]-ypre[24,:16])**2)/np.sum((valid_y[24,:16]-np.mean(ypre[24,:16]))**2))-0.35*(np.sum((valid_y[24,16:]-ypre[24,16:])**2)/np.sum((valid_y[24,16:]-np.mean(ypre[24,16:]))**2))

score2 = 1-0.65*(np.sum((valid_y[1472,:16]-ypre[1472,:16])**2)/np.sum((valid_y[1472,:16]-np.mean(ypre[1472,:16]))**2))-0.35*(np.sum((valid_y[1472,16:]-ypre[1472,16:])**2)/np.sum((valid_y[1472,16:]-np.mean(ypre[1472,16:]))**2))

score3 = 1-0.65*(np.sum((valid_y[2208,:16]-ypre[2208,:16])**2)/np.sum((valid_y[2208,:16]-np.mean(ypre[2208,:16]))**2))-0.35*(np.sum((valid_y[2208,16:]-ypre[2208,16:])**2)/np.sum((valid_y[2208,16:]-np.mean(ypre[2208,16:]))**2))

score4 = np.mean((score1,score2,score3))

print(score4, score1, score2, score3)

#%%
# 模型保存
import time
path = "./model_log/model{}.h5".format(time.time())
print(path)
model.save(path)



#%%
# 按时间截取数据
def dataframe_cut(df, begin_time='', end_time=''):
    if begin_time != '':
        df = df.loc[df.TimeStample >= pd.to_datetime(begin_time, format='%Y-%m-%d %H:%M:%S')]
    if end_time != '':
        df = df.loc[df.TimeStample <= pd.to_datetime(end_time, format='%Y-%m-%d %H:%M:%S')] 
    return df.reset_index(drop=True)

# 格式化日期
Qi_test1['TimeStample'] = pd.to_datetime(Qi_test1['TimeStample'], format ='%Y-%m-%d %H:%M:%S')
#%%
# 预测
testData1=Qi_test1.copy().reset_index(drop=True)
#%%
test1=dataframe_cut(testData1, begin_time='{}-01-4 02:00:00'.format('2019'), end_time='{}-01-31 23:00:00'.format('2019'))
test1=test1.loc[:,features].values.reshape((1, 4*7*8, n_features))
#%%
test2=dataframe_cut(testData1, begin_time='{}-03-4 02:00:00'.format('2019'), end_time='{}-03-31 23:00:00'.format('2019'))
test2=test2.loc[:,features].values.reshape((1, 4*7*8, n_features))
#%%
test3=dataframe_cut(testData1, begin_time='{}-05-4 02:00:00'.format('2019'), end_time='{}-05-31 23:00:00'.format('2019'))
test3=test3.loc[:,features].values.reshape((1, 4*7*8, n_features))
#%%
test4=dataframe_cut(testData1, begin_time='{}-07-4 02:00:00'.format('2019'), end_time='{}-07-31 23:00:00'.format('2019'))
test4=test4.loc[:,features].values.reshape((1, 4*7*8, n_features))
#%%
test5=dataframe_cut(testData1, begin_time='{}-10-4 02:00:00'.format('2019'), end_time='{}-10-31 23:00:00'.format('2019'))
test5=test5.loc[:,features].values.reshape((1, 4*7*8, n_features))

#%%
testData = np.vstack((test1,test2,test3,test4,test5))
#%%
yhat = model.predict(testData)



#%%
# 将结果写入CSV文件
submit = pd.read_csv('./final_data/submission.csv',index_col=0)
for i in range(len(yhat)):
    submit.iloc[i] = yhat[i]
#%%
submit.to_csv("sub_final1.csv")


