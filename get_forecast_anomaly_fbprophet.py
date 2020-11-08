#! /usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy
# import operator
import os
from fbprophet import Prophet

#%%
def get_feature_fbprophet(df_raw, ts_label='Qi', begin_time='', end_time=''):
    """
    :param df: DataFrame, 进行时序分解预测的数据集，即时间解析过的时序数据
    :param begin_time, end_time: str, df的起\终时间, '2018-02-07 00:00:00'
    """
    df = dataframe_cut(df_raw, begin_time=begin_time, end_time=end_time)
    forecast = get_forecast(df, ts_label=ts_label)
    detected = detect_anomalies(forecast)
    return detected


def get_forecast(df, ts_label='', interval_width=0.7,
                 changepoint_range=0.9):  # , interval_width = 0.99, changepoint_range = 0.8
    """
    时序预测 防止过拟合：1.变点选择范围100%但使用默认的稀疏先验；2.设置周期项强度0.7，适应更大的周期性波动
    :param df: DataFrame, 进行时序分解预测的数据集，即时间解析过的时序数据
    :param ts_label: str, 待预测的时序数据的标签
    :return: 时序分解后的数据：['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
       'daily', 'daily_lower', 'daily_upper', 'multiplicative_terms',
       'multiplicative_terms_lower', 'multiplicative_terms_upper', 'weekly',
       'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower',
       'yearly_upper', 'additive_terms', 'additive_terms_lower',
       'additive_terms_upper', 'yhat', 'fact']
    """
    # prophet模型有点傻 输入只能输入由两列构成的dataframe, 限制命名为['ds','y']
    test = df[['TimeStample', ts_label]].rename(columns={'TimeStample': 'ds', ts_label: 'y'})
    m = Prophet(yearly_seasonality=True, daily_seasonality=True
                # , yearly_seasonality = False, weekly_seasonality = False
                , seasonality_mode='multiplicative'
                , changepoint_prior_scale=0.01
                , interval_width=interval_width,
                changepoint_range=changepoint_range)
    #     m.add_country_holidays(country_name='CN')
    m.fit(test)
    future = m.make_future_dataframe(periods=168, freq='H')
    forecast = m.predict(future)
    forecast['fact'] = df[ts_label].reset_index(drop=True)
    m.plot(forecast)
    #     m.plot_components(forecast)
    return forecast


def detect_anomalies(forecast):
    """
    :param forecast 由fbprophet得到的时序预测结果dataframe,并将真实值存于'fact'字段
    :return: ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'daily', 'weekly',
       'fact', 'anomaly', 'delta_rate']
    """
    detected = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'daily', 'weekly', 'fact']].copy()

    # 是否异常的字段,0 正常,1真实值大于预测的上界,-1真实值小于预测的下界.
    detected['anomaly'] = 0
    detected.loc[detected['fact'] > detected['yhat_upper'], 'anomaly'] = 1
    detected.loc[detected['fact'] < detected['yhat_lower'], 'anomaly'] = -1

    # anomaly delta_rate
    detected['delta_rate'] = 0
    detected.loc[detected['anomaly'] == 1, 'delta_rate'] = \
        (detected['fact'] - detected['yhat_upper']) / forecast['fact']
    detected.loc[detected['anomaly'] == -1, 'delta_rate'] = \
        (detected['yhat_lower'] - detected['fact']) / forecast['fact']

    return detected


#     return detected['anomaly']
#     return detected.query('importance!=0').sort_values(by='importance',ascending=False)


'''
读取数据并处理时间戳
'''
def get_all_dataframe(file_path):
    df_list = []
    files = os.listdir(file_path)  # 得到文件夹下的所有文件名称
    print('list of data:', files)

    for file in files[1:]:
        if not os.path.isdir(file):
            df = pd.read_excel(file_path + '/' + file)
            print('=={}==========='.format(file))
            print(df.head())
            print(df.tail())
            print('The lenth is: %d' % len(df))
            df_list.append(df)
    return df_list


def parse_all_dataframe(df_list):
    for df in df_list:
        try:
            df.TimeStample = pd.to_datetime(df.TimeStample, format='%Y-%m-%d %H:%M:%S')
        except:
            df.TimeStample = pd.to_datetime(df.TimeStample, format='%Y-%m-%d')
    return df_list


# 按时间截取数据 TimeStample pd.to_datetime parsed
def dataframe_cut(df, begin_time='', end_time='', formats='%Y-%m-%d %H:%M:%S', drop=True):
    if begin_time != '':
        df = df.loc[df.TimeStample >= pd.to_datetime(begin_time, format=formats)]
    if end_time != '':
        df = df.loc[df.TimeStample <= pd.to_datetime(end_time, format=formats)]
    return df.reset_index(drop=drop)


def main():
    file_name = './data/入库流量数据.xlsx'
    df_raw = pd.read_excel(file_name)

    begin_time = '2018-01-01 00:00:00'
    end_time = '2018-02-07 00:00:00'
    print('begin_time is {}, end time is {}. Loading...'.format(begin_time, end_time))
    print(get_feature_fbprophet(df_raw, ts_label='Qi', begin_time=begin_time, end_time=end_time))
    # feature_detected = get_feature_fbprophet(df_list[0], ts_label='Qi', begin_time=begin_time, end_time=end_time)
#     merge and loop TODO


#%%
if __name__ == if __name__ == "__main__":
    main()