import pandas as pd
import numpy as np
import time
from matplotlib import pyplot
from models import *


def clean_data(train_agg, test_agg, train_log, test_log, train_flg, test_flg):
	print('start cleanning data up...................................')
	agg = pd.concat([train_agg, test_agg], copy=False)
	log = pd.concat([train_log, test_log], copy=False)

	log['EVT_LBL_1'] = log['EVT_LBL'].apply(lambda x: x.split('-')[0])
	log['EVT_LBL_2'] = log['EVT_LBL'].apply(lambda x: x.split('-')[1])
	log['EVT_LBL_3'] = log['EVT_LBL'].apply(lambda x: x.split('-')[2])

	log['OCC_TIM_time'] = log['OCC_TIM'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))

	# 用户唯一标识

	test_flg['FLAG'] = -1
	del test_flg['RST']

	flg = pd.concat([train_flg, test_flg], copy=False)
	data = pd.merge(agg, flg, on=['USRID'], how='left', copy=False)
	print(log['OCC_TIM'][:10])
	log['hour'] = log['OCC_TIM'].apply(lambda x: int(x.split(' ')[1].split(':')[0]))
	hour = log[['USRID', 'hour']].groupby(['USRID', 'hour']).size().reset_index().rename(columns={0: 'hour_cnt'})
	favorite_hour = hour.groupby('USRID').hour_cnt.max().reset_index().rename(columns={'hour_cnt': 'max_hour'})
	hour = hour.merge(favorite_hour, on='USRID', how='left')
	hour = hour[hour.hour_cnt == hour.max_hour]
	hour['favorite_hour'] = hour['hour']
	data = data.merge(hour[['USRID', 'favorite_hour']], on='USRID', how='left')
	
	data = data.fillna(method='pad')

	print("starting creating clicking information ...................")
	log_user = set(log['USRID'])
	data['is_click'] = data['USRID'].apply(lambda x: 1 if x in log_user else 0)
	return data, log


def create_time_feature(data, log):  # 构建时间特征
	# 这个部分将时间转化为秒，之后计算用户下一次的时间差特征
	# 这个部分可以发掘的特征其实很多很多很多很多
	print('starting calculate difference of time ....................')
	log = log.sort_values(['USRID', 'OCC_TIM_time'])
	log['difference_time'] = log.groupby(['USRID'])['OCC_TIM_time'].diff(-1).apply(np.abs)

	print("starting calculate time's statistic feature..................")
	log = log.groupby(['USRID'], as_index=False)['difference_time'].agg({
		'difference_time_mean': np.mean,
		'difference_time_std': np.std,
		'difference_time_min': np.min,
		'difference_time_max': np.max
	})
	
	data = pd.merge(data, log, on=['USRID'], how='left', copy=False)
	return data


def create_cnt_feature(data, log):  # 创建统计特征
	print("starting count the click number by USER......................")
	t1 = log[['USRID']].groupby('USRID').size().reset_index().rename(columns={0: 'user_clic_cnt'})
	data = data.merge(t1, on='USRID', how='left')
	data['user_clic_cnt'].fillna(0, inplace=True)
	return data


def create_event_feature(data, log):
	# -------------------------统计用户APP，WEB以及点击的券行为的特征-----------------------------#
	print('count user click feature on TCH_TYP ........................')
	t1 = log[['USRID', 'TCH_TYP']].groupby(['USRID', 'TCH_TYP']).size().reset_index().rename(
		columns={0: 'user_tch_cnt'})
	for i in range(3):
		data = data.merge(t1[t1.TCH_TYP == i][['USRID', 'user_tch_cnt']], on='USRID', how='left')

	print('count app_page label click number ..........................')
	t1 = log[['USRID', 'EVT_LBL_1']].groupby(['USRID', 'EVT_LBL_1']).size().reset_index().rename(
		columns={0: 'user_evt1_cnt'})
	for i in set(log['EVT_LBL_1']):
		data = data.merge(t1[t1.EVT_LBL_1 == i][['USRID', 'user_evt1_cnt']], on='USRID', how='left') 
	return data


def create_week_feature(data, log, label, start_day, end_day):

	# ----------------------------------统计用户按周划分的点击量，点击率，最大点击量-----------------------------------------
	print('count the %s week click number.......................' % label)
	
	log['day'] = log['OCC_TIM'].apply(lambda x: int(x.split(' ')[0].split('-')[2]))
	t1 = log[(log.day > start_day) & (log.day < end_day)][['USRID']].groupby('USRID').size().reset_index().rename(columns={0: '%s_week_cnt' % label})
	data = data.merge(t1, on='USRID', how='left')
	print('calculate the %s week click rate.......................' % label)
	data['%s_week_user_click_rate' % label] = data.apply(
		lambda x: 0 if x['user_clic_cnt'] == 0 else float(x['%s_week_cnt' % label]) / float(x['user_clic_cnt']), axis=1)

	print('count the %s week repeat click on the same label.......................' % label)
	t1 = log[(log.day > start_day) & (log.day < end_day)][['USRID', 'EVT_LBL_1']].groupby(
		['USRID', 'EVT_LBL_1']).size().reset_index().rename(columns={0: '%s_week_repeat_click' % label})

	print('count the %s week max click on the same label.......................' % label)
	t2 = t1.groupby('USRID')['%s_week_repeat_click' % label].max().reset_index().rename(
		columns={'%s_week_repeat_click' % label: '%s_week_max_click' % label})

	data = data.merge(t2[['USRID', '%s_week_max_click' % label]], on='USRID', how='left')
	return data


def calculate_week_difference(data):
	data = data.fillna(method='pad')
	print('calculating first and second week  feature ........................')
	data['diff_click_between_first_and_second'] = data.apply(lambda x: x['first_week_cnt']-x['second_week_cnt'], axis=1)
	data['diff_rate_between_first_and_second'] = data.apply(lambda x: float(x['first_week_user_click_rate'] - x['second_week_user_click_rate']), axis=1)
	data['diff_max_clck_between_first_and_second'] = data.apply(lambda x: x['first_week_max_click'] - x['second_week_max_click'], axis=1)
	
	print('calculating second and third week  feature ........................')
	data['diff_click_between_second_and_third'] = data.apply(lambda x: x['second_week_cnt']-x['third_week_cnt'], axis=1)
	data['diff_rate_between_second_and_third'] = data.apply(lambda x: float(x['second_week_user_click_rate'] - x['third_week_user_click_rate']), axis=1)
	data['diff_max_clck_between_second_and_third'] = data.apply(lambda x: x['second_week_max_click'] - x['third_week_max_click'], axis=1)
	
	print('calculating third and last week  feature ........................')
	data['diff_click_between_third_and_last'] = data.apply(lambda x: x['third_week_cnt'] - x['last_week_cnt'], axis=1)
	data['diff_rate_between_third_and_last'] = data.apply(lambda x: float(x['third_week_user_click_rate'] - x['last_week_user_click_rate']), axis=1)
	data['diff_max_clck_between_third_and_last'] = data.apply(lambda x: x['third_week_max_click'] - x['last_week_max_click'], axis=1)
	
	return data


def last_click_feature(data, log):
	# ------------------------统计这个用户最近一次与第一次点击的时间-------------------------
	print('create last click daytime...................')
	t1 = log[['USRID', 'day']].groupby('USRID')['day'].agg({'last_click_day': np.max}).reset_index()
	data = data.merge(t1[['USRID', 'last_click_day']], on='USRID', how='left')

	print('creat first click daytime..................')
	t1 = log[['USRID', 'day']].groupby('USRID')['day'].agg({'first_click_day': np.min}).reset_index()
	data = data.merge(t1[['USRID', 'first_click_day']], on='USRID', how='left')
	return data


def favorite_label(data, log):
	# ------------------------统计这个用户最喜爱的LBL-----------------------------------
	print('find USER favorite app_page label....................................')
	t1 = log[['USRID', 'EVT_LBL_1']].groupby(['USRID', 'EVT_LBL_1']).size().reset_index().rename(
		columns={0: 'user_evt1_cnt'})
	t2 = t1.groupby('USRID').user_evt1_cnt.max().reset_index().rename(columns={'user_evt1_cnt': 'max_user_evt1_cnt'})
	t1 = t1.merge(t2, on='USRID', how='left')
	t1 = t1[t1.user_evt1_cnt == t1.max_user_evt1_cnt]
	t1['love_LBl'] = t1['EVT_LBL_1']
	data = data.merge(t1[['USRID', 'love_LBl']], on='USRID', how='left')
	data = data.fillna(0)
	data['love_LBl'] = data['love_LBl'].apply(lambda x: int(x))
	return data


def create_pre_week_features(data):
	first_three_week_data = data.drop(['last_week_cnt', 'last_week_user_click_rate', 'last_week_max_click'], axis=1)
	last_three_week_data = data.drop(['first_week_cnt', 'first_week_user_click_rate', 'first_week_max_click'], axis=1)
	first_three_week_data = first_three_week_data.values
	last_three_week_data = last_three_week_data.values
	
	last_week_cnt = data['last_week_cnt'].values
	new_week_cnt = xgboost_for_feature(first_three_week_data, last_week_cnt, last_three_week_data)
	data['new_week_cnt'] = list(new_week_cnt)
	
	last_week_user_click_rate = data['last_week_user_click_rate'].values
	new_week_user_click_rate = xgboost_for_feature(first_three_week_data, last_week_user_click_rate, last_three_week_data)
	data['new_week_user_click_rate'] = list(new_week_user_click_rate)
	
	last_week_max_click = data['last_week_max_click'].values
	new_week_max_click = xgboost_for_feature(first_three_week_data, last_week_max_click, last_three_week_data)
	data['new_week_max_click'] = list(new_week_max_click)
	
	return data


def create_new_week_feature(data1, data):
	data['new_week_cnt'] = data1['new_week_cnt']
	data['new_week_user_click_rate'] = data1['new_week_user_click_rate']
	data['new_week_max_click'] = data1['new_week_max_click']
	return data

	
def add_time_feature(data, log):
	print("adding time feature..................................")
	log = log.sort_values(['USRID', 'OCC_TIM_time'])
	last_click_time = log[['USRID', 'OCC_TIM_time']].groupby('USRID')['OCC_TIM_time'].agg({'last_click_time': np.max}).reset_index()
	data = data.merge(last_click_time, on='USRID', how='left')
	print(last_click_time.info())
	
	data['max_next_click_time'] = data.apply(lambda x: 0 if x['next_time_mean'] == 0 else int(x['last_click_time']) + int(x['next_time_max']), axis=1)
	data['mean_next_click_time'] = data.apply(lambda x: 0 if x['next_time_mean'] == 0 else int(x['last_click_time'] + x['next_time_mean']), axis=1)
	# data.fillna()
	# data.to_csv('./data_transfer/next_click_time.csv', index=None)
	# data['min_next_click_time'] = data.apply(lambda x: float(x['last_click_time'] + x['next_time_min']), axis=1)

	data['max_next_time_is_in_new_week'] = data['max_next_click_time'].apply(lambda x: 1 if (x > 1522512000) & (x < 1523116800) else 0)
	data['mean_next_time_is_in_new_week'] = data['mean_next_click_time'].apply(lambda x: 1 if (x > 1522512000) & (x < 1523116800) else 0)
	
	print(data[:10])
	data = data.drop(['max_next_click_time', 'mean_next_click_time'], axis=1)
	# print(data[:10])
	return data


def drop_unuseful_feature(data):
	data = data.drop(['first_week_cnt', 'first_week_user_click_rate', 'first_week_max_click'], axis=1)
	data = data.drop(['second_week_cnt', 'second_week_user_click_rate', 'second_week_max_click'], axis=1)
	data = data.drop(['third_week_cnt', 'third_week_user_click_rate', 'third_week_max_click'], axis=1)
	
	data = data.drop(['diff_click_between_first_and_second', 'diff_rate_between_first_and_second', 'diff_max_clck_between_first_and_second'], axis=1)
	data = data.drop(['diff_click_between_second_and_third', 'diff_rate_between_second_and_third', 'diff_max_clck_between_second_and_third'], axis=1)
	data = data.drop(['diff_click_between_third_and_last', 'diff_rate_between_third_and_last', 'diff_max_clck_between_third_and_last'], axis=1)
	
	return data