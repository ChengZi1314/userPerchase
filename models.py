# coding:utf-8
import os
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.cross_validation import train_test_split
import xgboost as xgb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lgb_model(X, y, test):
	print('///////////////////////////', X.dtype)
	N = 5
	skf = StratifiedKFold(n_splits=N,shuffle=True,random_state=42)

	xx_cv = []
	xx_pre = []

	# specify your configurations as a dict
	params = {
				      	'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 15,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 8,
        'verbose': 0,
        #'is_unbalance':True,
	    # 'lambda_l1':0.5,
    	# 'lambda_l2':35,
        #'scale_pos_weight':24
	}

	for train_in, test_in in skf.split(X, y):
		X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]
		lgb_train = lgb.Dataset(X_train, y_train)
		lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

		print('Start training...')
		gbm = lgb.train(params, lgb_train, num_boost_round=40000,
						valid_sets=lgb_eval, verbose_eval=250, early_stopping_rounds=50)

		print('Start predicting...')
		y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
		xx_cv.append(roc_auc_score(y_test,y_pred))
		xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))
	return xx_cv, xx_pre


# def auc(y_true, y_pred):
#
# 	ptas = tf.stack([binary_pta(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
# 	pfas = tf.stack([binary_pfa(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
# 	pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
# 	binSizes = -(pfas[1:]-pfas[:-1])
# 	s = ptas*binSizes
# 	return K.sum(s, axis=0)
#
#
# def binary_pfa(y_true, y_pred, threshold=K.variable(value=0.5)):
#
# 	y_pred = K.cast(y_pred >= threshold, 'float32')
# 	# N = total number of negative labels
# 	N = K.sum(1 - y_true)
# 	# FP = total number of false alerts, alerts from the negative class labels
# 	FP = K.sum(y_pred - y_pred * y_true)
# 	return FP/N
#
#
# def binary_pta(y_true, y_pred, threshold=K.variable(value=0.5)):
#
# 	y_pred = K.cast(y_pred >= threshold, 'float32')
# 	# P = total number of positive labels
# 	P = K.sum(y_true)
# 	# TP = total number of correct alerts, alerts from the positive class labels
# 	TP = K.sum(y_pred * y_true)
# 	return TP/P
#
#
# def nn_train(train_x, train_y, submit):
# 	# 预处理矩阵，将训练数据变成按条来分割的数据，标记数据编码成one-hot
# 	# train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
#
# 	scaler = MinMaxScaler(feature_range=(0, 1))
# 	train_x = scaler.fit(train_x)
# 	test = scaler.fit(submit)
# 	model = Sequential()
# 	model.add(Dense(512, input_shape=(train_x.shape[1],)))  # 输入维度, 512==输出维度
# 	model.add(Activation('relu'))  # 激活函数
# 	model.add(Dropout(0.5))  # dropout<br><br>#第二层
# 	model.add(Dense(1))
# 	model.add(Activation('softmax'))
# 	adam = Adam(lr=0.001, beta_1=0.9, epsilon=1e-8, decay=0.0)
# 	model.compile(optimizer=adam, loss='mse', metrics=['mse'])
# 	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
# 	model.fit(train_x, train_y, epochs=2000, batch_size=32, callbacks=[early_stopping], verbose=1, validation_split=0.3)
# 	submit_answer = model.predict(test)
# 	return submit_answer


def xgboost(X, y, test):
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1729)
	xlf = xgb.XGBRegressor(	max_depth=10,
							learning_rate=0.01,
							n_estimators=2000,
							silent=True,
							objective='reg:logistic',
							nthread=-1,
							gamma=0,
							min_child_weight=1,
							max_delta_step=0,
							subsample=0.85,
							colsample_bytree=0.7,
							colsample_bylevel=1,
							reg_alpha=0,
							reg_lambda=1,
							scale_pos_weight=1,
							seed=1440,
							missing=None)
	xlf.fit(x_train, y_train, eval_metric='auc', verbose=True, eval_set=[(x_test, y_test)], early_stopping_rounds=100)
	y_pred = xlf.predict(x_test, ntree_limit=xlf.best_ntree_limit)
	auc_score = roc_auc_score(y_test, y_pred)
	predictions = xlf.predict(test)
	# count = 0
	# a = []
	# for t in range(200):
	# 	for i in range(len(y_pred)):
	# 		if y_pred[i] < t*0.0001:
	# 			y_pred[i] = 0.000001
	# 			count += 1
	# 	print('when t is %d' % t, 'the count is ', count / len(y_pred))
	# 	count = 0
	# 	print(roc_auc_score(y_test, y_pred))
	# 	a.append(roc_auc_score(y_test, y_pred))
	# threshold = a.index(max(a)) * 200
	#
	# for i in range(len(predictions)):
	# 	if predictions[i] < threshold:
	# 		predictions[i] = 0.000001
	
	return predictions, auc_score


def xgboost_for_feature(X, y, test):
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1729)
	xlf = xgb.XGBRegressor(max_depth=10,
	                       learning_rate=0.01,
	                       n_estimators=500,
	                       silent=True,
	                       objective='reg:linear',
	                       nthread=-1,
	                       gamma=0,
	                       min_child_weight=1,
	                       max_delta_step=0,
	                       subsample=0.85,
	                       colsample_bytree=0.7,
	                       colsample_bylevel=1,
	                       reg_alpha=0,
	                       reg_lambda=1,
	                       scale_pos_weight=1,
	                       seed=1440,
	                       missing=None)
	xlf.fit(x_train, y_train, eval_metric='rmse', verbose=True, eval_set=[(x_test, y_test)], early_stopping_rounds=50)
	predictions = xlf.predict(test, ntree_limit=xlf.best_ntree_limit)
	
	return predictions


def lgb_for_feature(X,y,test):
	N = 5
	skf = StratifiedKFold(n_splits=N,shuffle=True,random_state=42)

	xx_cv = []
	xx_pre = []

	# specify your configurations as a dict
	params = {
				'boosting_type': 'gbdt',
				'objective': 'regression',
				'metric': {'rmse'},
				'num_leaves': 32,
				'learning_rate': 0.01,
				'feature_fraction': 0.9,
				'bagging_fraction': 0.8,
				'bagging_freq': 5,
				'verbose': 1,
				'lambda_l1': 0.5,
				'lambda_l2': 35,
	}

	for train_in, test_in in skf.split(X, y):
		X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]
		lgb_train = lgb.Dataset(X_train, y_train)
		lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

		print('Start training...')
		gbm = lgb.train(params, lgb_train, num_boost_round=2000,
						valid_sets=lgb_eval, verbose_eval=100, early_stopping_rounds=50)

		print('Start predicting...')
		y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
		xx_cv.append(mean_squared_error(y_test, y_pred))
		xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))
	prediction = xx_pre[xx_cv.index(max(xx_cv))]
	return prediction
