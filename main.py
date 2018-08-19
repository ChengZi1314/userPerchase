from preProcessing import *
from models import *
from sklearn.grid_search import GridSearchCV

ONLINE = True

train_agg = pd.read_csv('./data/train_agg.csv', sep='\t')
test_agg = pd.read_csv('./data/test_agg.csv', sep='\t')
train_log = pd.read_csv('./data/train_log.csv', sep='\t')
test_log = pd.read_csv('./data/test_log.csv', sep='\t')
train_flg = pd.read_csv('./data/train_flg.csv', sep='\t')
test_flg = pd.read_csv('./data/submit_sample.csv', sep='\t')

data, log = clean_data(train_agg, test_agg, train_log, test_log, train_flg, test_flg)

print(data.info())
data = create_time_feature(data, log)

data = create_cnt_feature(data, log)

data = create_event_feature(data, log)

data = create_week_feature(data, log, 'first', 3, 11)
data = create_week_feature(data, log, 'second', 10, 18)
data = create_week_feature(data, log, 'third', 17, 25)
data = create_week_feature(data, log, 'last', 24, 32)

data = calculate_week_difference(data)

data = last_click_feature(data, log)

data = favorite_label(data, log)

data = create_pre_week_features(data)

data = drop_unuseful_feature(data)

data = add_time_feature(data, log)

print(data.info())


train = data[data['FLAG'] != -1]
test = data[data['FLAG'] == -1]
train_userid = train.pop('USRID')

y = train.pop('FLAG')
col = train.columns
X = train[col].values
test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test = test[col].values


auc_score, predictions = lgb_model(X, y, test)
# pred_xgboost, auc_xgboost = xgboost(X, y, test)
print('the final lgb predict auc', auc_score)
# print('the final xgboost pred auc', auc_xgboost)
# res = pd.DataFrame()
# res['USRID'] = list(test_userid.values)
# for i in range(5):
# 	res['lgb_%d' % i] = list(predictions[i])
# res[]



# if ONLINE:
# 	import time
#
# 	s = []
# 	for i in predictions:
# 		s.append(i)
# 	res = pd.DataFrame()
# 	res['USRID'] = list(test_userid.values)
# 	res['RST'] = list(s[auc_score.index(max(auc_score))])
#
# 	time_date = time.strftime('%m-%d', time.localtime(time.time()))
# 	res.to_csv('./history_submit/%5f_%s.csv' % (float(np.max(auc_score)), str(time_date)), index=False, sep='\t')


# original [0.85186000090305014, 0.86137105108703738, 0.87296813717246347, 0.85274275866883709, 0.86223233028706414]
# time_feature 改成非abs [0.85340515716833598, 0.85928675799344867, 0.87316335164788073, 0.85187475144778568, 0.862196528404507]
# add_difference [0.845104151772612, 0.85875941978194248, 0.86525377363619083, 0.83792393550874988, 0.8535010048552063]
# add new_week_feature [0.85285091019919779, 0.8615201848769104, 0.87338277950176402, 0.85332731861706368, 0.86096866515601567]
# add week feature [0.85362344129798395, 0.86144444036202861, 0.87272642735746209, 0.85080413383610498, 0.8621157386300522]
# 将差分的时间差加到最后点击时间上的特征 [0.85254801216107401, 0.8611680765132651, 0.87296507516935007, 0.85132651156725847, 0.86210848403806029]
# 获取到下一周是否点击 [0.85361953748786523, 0.86178609146055962, 0.87278333350763204, 0.85150453171750018, 0.86182951200055546]

# 加入hour信息  [0.84618188556850771, 0.85742852082456011, 0.864464436341285, 0.84150092043813596, 0.85283655017958881]
# 剔除hour信息 [0.84729094391306625, 0.85774412296990088, 0.86321937876761812, 0.84144589859757435, 0.8525268167877289]     [0.84582245283444824, 0.8578135554418761, 0.86458371313948879, 0.84175077989219305, 0.85258268656761416]
# 改变差分的abs为正常值[0.846025, 0.857234, 0.86528, 0.842181, 0.850743]
# [0.84730881677866998, 0.86569823912844823, 0.86634752114831337, 0.83941150372900164, 0.85817004141806741]\
# [0.85411551687312093, 0.84953011534467116, 0.86519756015758209, 0.86083672990275228, 0.85182084733677199]
# [0.85161211374112822, 0.85012433452220926, 0.86415691096328695, 0.86035853702905973, 0.85071859745023892]
# [0.84702286216941358, 0.85020903547403615, 0.86698590525411023, 0.85949936487411271, 0.8508342737501513]
# [0.91455406480130319, 0.88361765465339415, 0.91209939971700427, 0.91639447298289034, 0.90982883670155246]
# [0.86499849542059237, 0.87253631137679899, 0.88253973004021646, 0.87163330122922256, 0.87024942942796313]
