import os
import time
import numpy
import pickle
import pandas
import sys
import tensorflow as tf
import random as rn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from math import sqrt
from numpy import concatenate
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import plot_model
from keras.models import load_model
from keras.layers import Dropout, Dense, Conv1D , MaxPooling1D, LSTM, Flatten, SimpleRNN,ATSLSTM
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from hyperopt import Trials, STATUS_OK, tpe
from hyperopt.plotting import main_plot_history
from hyperas import optim
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import choice, uniform, qloguniform, randint, qlognormal, lognormal
from collections import OrderedDict

seed = 7
np.random.seed(seed)
matplotlib.use('agg')

def load_dataset(datasource1: str, datasource2: str, datasource3: str, datasource4: str) -> (numpy.ndarray, MinMaxScaler):
    dataframe1 = pandas.read_csv(datasource1, usecols=[1])
    dataframe1 = dataframe1.fillna(method='pad')
    dataset1 = dataframe1.values
    dataset1 = dataset1.astype('float32')
    dataset1 = dataset1[0:50]

    dataframe2 = pandas.read_csv(datasource2, usecols=[1])
    dataframe2 = dataframe2.fillna(method='pad')
    dataset2 = dataframe2.values
    dataset2 = dataset2.astype('float32')
    dataset2 = dataset2[0:50]

    dataframe3 = pandas.read_csv(datasource3, usecols=[1])
    dataframe3 = dataframe3.fillna(method='pad')
    dataset3 = dataframe3.values
    dataset3 = dataset3.astype('float32')
    dataset3 = dataset3[0:50]

    dataframe4 = pandas.read_csv(datasource4, usecols=[1])
    dataframe4 = dataframe4.fillna(method='pad')
    dataset4 = dataframe4.values
    dataset4 = dataset4.astype('float32')

    dataset = numpy.concatenate((dataset1, dataset2, dataset3, dataset4), axis=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset, scaler


def create_dataset(dataset: numpy.ndarray, look_back: int=1) -> (numpy.ndarray, numpy.ndarray):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return numpy.array(data_x), numpy.array(data_y)


datasource5 = r'.\data\rul\5-capacity168.csv'
datasource6 = r'.\data\rul\6-capacity168.csv'
datasource7 = r'.\data\rul\7-capacity168.csv'
datasource18 = r'.\data\rul\18-capacity132.csv'

dataset, scaler = load_dataset(datasource5, datasource6, datasource18, datasource7)
joblib.dump(scaler, r'.\result\scaler_rul_TPE.pickle')

look_back = 30
dataset_x, dataset_y = create_dataset(dataset, look_back)
dataset_x = numpy.concatenate((dataset_x[0:20], dataset_x[50:70], dataset_x[100:120], dataset_x[150:]), axis=0)
dataset_y = numpy.concatenate((dataset_y[0:20], dataset_y[50:70], dataset_y[100:120], dataset_y[150:]), axis=0)
data_concatenate = numpy.concatenate((dataset_x, dataset_y.reshape(dataset_y.shape[0], 1)), axis=1)
numpy.random.shuffle(data_concatenate)
train_size = round(data_concatenate.shape[0] * 0.9)
train_data = data_concatenate[0: train_size]
vaild_data = data_concatenate[train_size:]
x_train = train_data[:, 0: -1]
y_train = train_data[:, -1]
x_vaild = vaild_data[:, 0: -1]
y_vaild = vaild_data[:, -1]
x_train = numpy.reshape(x_train, (x_train.shape[0], 10, 3))
x_vaild = numpy.reshape(x_vaild, (x_vaild.shape[0], 10, 3))

space = OrderedDict([('layer',hp.qnormal('layer', 1.2, 0.22, 1)),
					 ('units1',hp.qlognormal('units1', np.log(40), np.log(1.33), 1)),
                    ('units2',hp.qlognormal('units2', np.log(40), np.log(1.33), 1)),
					 ('units3',hp.qlognormal('units3', np.log(40), np.log(1.33), 1)),
                    ('learning rate', hp.lognormal('learning rate', np.log(0.0007), np.log(1.3))),
                    ('batch_size', hp.qlognormal('batch_size', np.log(22), np.log(1.3), 1)),
                    ('nb_epochs', hp.qlognormal('nb_epochs', np.log(98), np.log(1.3), 1)),
					('filters', hp.qlognormal('filters', np.log(70), np.log(1.25), 1)),
					('kernel_size', hp.qlognormal('kernel_size', np.log(4.5), np.log(1.3), 1)),
					('strides', hp.qlognormal('strides', np.log(3.6), np.log(1.2), 1)),
					('pool_size', hp.qnormal('pool_size', 1.4, 0.21, 1)),
					 ('dropout', hp.lognormal('dropout', np.log(0.05), np.log(1.3)))
                    ])


def f_nn(params):
	model = Sequential()
	model.add(Conv1D(filters=np.int(params['filters']), kernel_size=np.int(params['kernel_size']), strides=np.int(params['strides']), padding='same', activation='relu', input_shape=(10, 3)))
	model.add(MaxPooling1D(pool_size=np.int(params['pool_size']), padding='valid'))
	layer = np.int(params['layer'])
	if layer == 1:
			model.add(ATSLSTM(np.int(params['units1']), return_sequences=False))
	if layer == 2:
			model.add(ATSLSTM(np.int(params['units2']), return_sequences=True))
			model.add(ATSLSTM(np.int(params['units3']), return_sequences=False))
	model.add(Dropout(params['dropout']))
	model.add(Dense(1))
	optimizer = Adam(lr=params['learning rate'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	model.fit(x_train, y_train, epochs=np.int(params['nb_epochs']), batch_size=np.int(params['batch_size']), verbose=1, shuffle=False)
	mse = model.evaluate(x_vaild, y_vaild, verbose=0)
	print('Train mse:', mse)
	return {'loss': mse, 'status': STATUS_OK, 'model': model}


trials = Trials()
_ = fmin(f_nn, space, algo=tpe.suggest, max_evals=1, trials=trials)

best_loss = trials.best_trial['result']['loss']
best_params = trials.best_trial['misc']['vals']

hyperopt = []
result = []
for i in range(len(trials)):
	hyperopt.append(trials.trials[i].get('misc').get('vals'))
for j in range(len(trials)):
	result.append(trials.results[j].get('loss'))

with open(r'./result/process_data_rul_TPE.csv', 'a', encoding='utf-8') as f:
	for k in range(len(hyperopt)):
		f.write(str(hyperopt[k]) + "\n")

with open(r'./result/process_result_rul_TPE.csv', 'a', encoding='utf-8') as f:
	for m in range(len(result)):
		f.write(str(result[m]) + "\n")

print('Best Validation LOSS: {}'.format(best_loss))
print('Best Params: {}'.format(best_params))