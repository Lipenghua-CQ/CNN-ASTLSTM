import sys
import tensorflow as tf
import random as rn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pickle
import matplotlib
import skopt

from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv1D , MaxPooling1D, LSTM, Flatten, SimpleRNN,ATSLSTM
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from math import sqrt
from numpy import concatenate
from hyperopt import Trials, STATUS_OK, tpe
from hyperopt.plotting import main_plot_history
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas import optim
from hyperas.distributions import choice, uniform, qloguniform, randint, qlognormal, lognormal
from collections import OrderedDict
from neptunecontrib.monitoring.utils import pickle_and_send_artifact

seed = 7
np.random.seed(seed)
matplotlib.use('agg')

file_name1 = './data/soh/vltm5.csv'
file_name2 = './data/soh/vltm6.csv'
file_name3 = './data/soh/vltm7.csv'
file_name4 = './data/soh/vltm18.csv'
file_name5 = './data/soh/vltm45.csv'
file_name6 = './data/soh/vltm46.csv'
file_name7 = './data/soh/vltm47.csv'
file_name8 = './data/soh/vltm48.csv'
file_name9 = './data/soh/vltm53.csv'
file_name10 = './data/soh/vltm54.csv'
file_name11 = './data/soh/vltm55.csv'
file_name12 = './data/soh/vltm56.csv'

series1 = read_csv(file_name1, header=None, parse_dates=[0], squeeze=True)
series2 = read_csv(file_name2, header=None, parse_dates=[0], squeeze=True)
series3 = read_csv(file_name3, header=None, parse_dates=[0], squeeze=True)
series4 = read_csv(file_name4, header=None, parse_dates=[0], squeeze=True)
series5 = read_csv(file_name5, header=None, parse_dates=[0], squeeze=True)
series6 = read_csv(file_name6, header=None, parse_dates=[0], squeeze=True)
series7 = read_csv(file_name7, header=None, parse_dates=[0], squeeze=True)
series8 = read_csv(file_name8, header=None, parse_dates=[0], squeeze=True)
series9 = read_csv(file_name9, header=None, parse_dates=[0], squeeze=True)
series10 = read_csv(file_name10, header=None, parse_dates=[0], squeeze=True)
series11 = read_csv(file_name11, header=None, parse_dates=[0], squeeze=True)
series12 = read_csv(file_name12, header=None, parse_dates=[0], squeeze=True)

raw_values5 = series1.values
raw_values6 = series2.values
raw_values7 = series3.values
raw_values18 = series4.values
raw_values45 = series5.values
raw_values46 = series6.values
raw_values47 = series7.values
raw_values48 = series8.values
raw_values53 = series9.values
raw_values54 = series10.values
raw_values55 = series11.values
raw_values56 = series12.values
raw_values = np.concatenate((raw_values5, raw_values6, raw_values7, raw_values18, raw_values45, raw_values46, raw_values47, raw_values48, raw_values53, raw_values54, raw_values55, raw_values56), axis=0)


def create_dataset(dataset):
	dataX, dataY = [], []
	look_back = 2640
	for i in range(0, len(dataset) - look_back, 2641):
		a = dataset[i:(i + look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	dataY = np.array(dataY)
	dataY = np.reshape(dataY, (dataY.shape[0], 1))
	for i in range(len(dataY)):
		if dataY[i] == 0:
			dataY[i] = dataY[i - 1]
	dataset = np.concatenate((dataX, dataY), axis=1)
	return dataset, dataY


dataset, dataY = create_dataset(raw_values)
dataset_5, dataY_5 = create_dataset(raw_values5)
dataset_6, dataY_6 = create_dataset(raw_values6)
dataset_7, dataY_7 = create_dataset(raw_values7)
dataset_18, dataY_18 = create_dataset(raw_values18)
dataset_45, dataY_45 = create_dataset(raw_values45)
dataset_46, dataY_46 = create_dataset(raw_values46)
dataset_47, dataY_47 = create_dataset(raw_values47)
dataset_48, dataY_48 = create_dataset(raw_values48)
dataset_53, dataY_53 = create_dataset(raw_values53)
dataset_54, dataY_54 = create_dataset(raw_values54)
dataset_55, dataY_55 = create_dataset(raw_values55)
dataset_56, dataY_56 = create_dataset(raw_values56)

train_size_5 = int(dataset_5.shape[0] * 0.6)
train_size_6 = int(dataset_6.shape[0] * 0.6)
train_size_7 = int(dataset_7.shape[0] * 0.6)
train_size_18 = int(dataset_18.shape[0] * 0.6)
train_size_45 = int(dataset_45.shape[0] * 0.6)
train_size_46 = int(dataset_46.shape[0] * 0.6)
train_size_47 = int(dataset_47.shape[0] * 0.6)
train_size_48 = int(dataset_48.shape[0] * 0.6)
train_size_53 = int(dataset_53.shape[0] * 0.6)
train_size_54 = int(dataset_54.shape[0] * 0.6)
train_size_55 = int(dataset_55.shape[0] * 0.6)
train_size_56 = int(dataset_56.shape[0] * 0.6)

valid_size_5 = int(dataset_5.shape[0] * 0.7)
valid_size_6 = int(dataset_6.shape[0] * 0.7)
valid_size_7 = int(dataset_7.shape[0] * 0.7)
valid_size_18 = int(dataset_18.shape[0] * 0.7)
valid_size_45 = int(dataset_45.shape[0] * 0.7)
valid_size_46 = int(dataset_46.shape[0] * 0.7)
valid_size_47 = int(dataset_47.shape[0] * 0.7)
valid_size_48 = int(dataset_48.shape[0] * 0.7)
valid_size_53 = int(dataset_53.shape[0] * 0.7)
valid_size_54 = int(dataset_54.shape[0] * 0.7)
valid_size_55 = int(dataset_55.shape[0] * 0.7)
valid_size_56 = int(dataset_56.shape[0] * 0.7)

train_5, vaild_5, test_5 = dataset_5[0:train_size_5], dataset_5[train_size_5:valid_size_5], dataset_5[valid_size_5:]
train_6, vaild_6, test_6 = dataset_6[0:train_size_6], dataset_6[train_size_6:valid_size_6], dataset_6[valid_size_6:]
train_7, vaild_7, test_7 = dataset_7[0:train_size_7], dataset_7[train_size_7:valid_size_7], dataset_7[valid_size_7:]
train_18, vaild_18, test_18 = dataset_18[0:train_size_18], dataset_18[train_size_18:valid_size_18], dataset_18[valid_size_18:]
train_45, vaild_45, test_45 = dataset_45[0:train_size_45], dataset_45[train_size_45:valid_size_45], dataset_45[valid_size_45:]
train_46, vaild_46, test_46 = dataset_46[0:train_size_46], dataset_46[train_size_46:valid_size_46], dataset_46[valid_size_46:]
train_47, vaild_47, test_47 = dataset_47[0:train_size_47], dataset_47[train_size_47:valid_size_47], dataset_47[valid_size_47:]
train_48, vaild_48, test_48 = dataset_48[0:train_size_48], dataset_48[train_size_48:valid_size_48], dataset_48[valid_size_48:]
train_53, vaild_53, test_53 = dataset_53[0:train_size_53], dataset_53[train_size_53:valid_size_53], dataset_53[valid_size_53:]
train_54, vaild_54, test_54 = dataset_54[0:train_size_54], dataset_54[train_size_54:valid_size_54], dataset_54[valid_size_54:]
train_55, vaild_55, test_55 = dataset_55[0:train_size_55], dataset_55[train_size_55:valid_size_55], dataset_55[valid_size_55:]
train_56, vaild_56, test_56 = dataset_56[0:train_size_56], dataset_56[train_size_56:valid_size_56], dataset_56[valid_size_56:]

train = np.concatenate((train_5, train_6, train_7, train_18, train_45, train_46, train_47, train_48, train_53, train_54, train_55, train_56), axis=0)
vaild = np.concatenate((vaild_5, vaild_6, vaild_7, vaild_18, vaild_45, vaild_46, vaild_47, vaild_48, vaild_53, vaild_54, vaild_55, vaild_56), axis=0)

np.random.shuffle(train)
features = train[:, :-1]
labels = train[:, -1]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train)
train = train.reshape(train.shape[0], train.shape[1])
vaild = vaild.reshape(vaild.shape[0], vaild.shape[1])
test_5 = test_5.reshape(test_5.shape[0], test_5.shape[1])
train_scaled = scaler.transform(train)
vaild_scaled = scaler.transform(vaild)
test5_scaled = scaler.transform(test_5)
joblib.dump(scaler, r'.\result\scaler_soh_TPE.pickle')
x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
x_vaild, y_vaild = vaild_scaled[:, 0:-1], vaild_scaled[:, -1]
x_test, y_test = test5_scaled[:, 0:-1], test5_scaled[:, -1]
x_train = x_train.reshape(x_train.shape[0], 660, 4)
x_vaild = x_vaild.reshape(x_vaild.shape[0], 660, 4)
x_test = x_test.reshape(x_test.shape[0], 660, 4)

space = OrderedDict([('layer',hp.qlognormal('layer', np.log(2), np.log(1.1), 1)),
					 ('units1',hp.qlognormal('units1', np.log(30), np.log(1.3), 1)),
                    ('units2',hp.qlognormal('units2', np.log(30), np.log(1.3), 1)),
					 ('units3',hp.qlognormal('units3', np.log(30), np.log(1.3), 1)),
                    ('units4',hp.qlognormal('units4', np.log(30), np.log(1.3), 1)),
					 ('units5',hp.qlognormal('units5', np.log(30), np.log(1.3), 1)),
                    ('units6',hp.qlognormal('units6', np.log(30), np.log(1.3), 1)),
                    ('learning rate', hp.lognormal('learning rate', np.log(0.0011), np.log(1.2))),
                    ('batch_size', hp.qlognormal('batch_size', np.log(10), np.log(1.5), 1)),
                    ('nb_epochs', hp.qlognormal('nb_epochs', np.log(110), np.log(1.2), 1)),
					('filters', hp.qlognormal('filters', np.log(40), np.log(1.15), 1)),
					('kernel_size', hp.qlognormal('kernel_size', np.log(7), np.log(1.12), 1)),
					('strides', hp.qnormal('strides', 4, 0.5, 1)),
					('pool_size', hp.qnormal('pool_size', 3, 0.5, 1)),
					 ('dropout', hp.uniform('dropout', 0.01, 0.1))
                    ])


def f_nn(params):
	model = Sequential()
	model.add(Conv1D(filters=np.int(params['filters']), kernel_size=np.int(params['kernel_size']), strides=np.int(params['strides']), padding='same', activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
	model.add(MaxPooling1D(pool_size=np.int(params['pool_size']), padding='valid'))
	layer = np.int(params['layer'])
	if layer == 1:
			model.add(ATSLSTM(np.int(params['units1']), return_sequences=False))
	if layer == 2:
			model.add(ATSLSTM(np.int(params['units2']), return_sequences=True))
			model.add(ATSLSTM(np.int(params['units3']), return_sequences=False))
	if layer == 3:
			model.add(ATSLSTM(np.int(params['units4']), return_sequences=True))
			model.add(ATSLSTM(np.int(params['units5']), return_sequences=True))
			model.add(ATSLSTM(np.int(params['units6']), return_sequences=False))
	model.add(Dropout(params['dropout']))
	model.add(Dense(1))
	optimizer = Adam(lr=params['learning rate'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	model.fit(x_train, y_train, epochs=np.int(params['nb_epochs']), batch_size=np.int(params['batch_size']), verbose=1, shuffle=False)
	mse = model.evaluate(x_vaild, y_vaild, verbose=0)
	print('Train mse:', mse)
	return {'loss': mse, 'status': STATUS_OK, 'model': model}


trials = Trials()
_ = fmin(f_nn, space, algo=tpe.suggest, max_evals=20, trials=trials)

best_loss = trials.best_trial['result']['loss']
best_params = trials.best_trial['misc']['vals']

hyperopt = []
result = []

for i in range(len(trials)):
	hyperopt.append(trials.trials[i].get('misc').get('vals'))
for j in range(len(trials)):
	result.append(trials.results[j].get('loss'))

with open(r'./result/process_data_soh_TPE.csv', 'a', encoding='utf-8') as f:
	for k in range(len(hyperopt)):
		f.write(str(hyperopt[k]) + "\n")

with open(r'./result/process_result_soh_TPE.csv', 'a', encoding='utf-8') as f:
	for m in range(len(result)):
		f.write(str(result[m]) + "\n")

# log metrics
print('Best Validation LOSS: {}'.format(best_loss))
print('Best Params: {}'.format(best_params))