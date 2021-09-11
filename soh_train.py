import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, ATSLSTM
from keras.utils import plot_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['PYTHONHASHSEED'] = '0'

seed = 7
np.random.seed(seed)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

loss_list = []

# scale train and test data to [-1, 1]
def scale(train, test):
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]


# fit an AST-LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 660, 4)
	model = Sequential()
	model.add(Conv1D(filters=46, kernel_size=7, strides=4, padding='same', activation='relu', input_shape=(X.shape[1], X.shape[2])))
	model.add(MaxPooling1D(pool_size=2, padding='valid'))
	model.add(ATSLSTM(24, return_sequences=True))
	model.add(ATSLSTM(28, return_sequences=False))
	model.add(Dropout(0.0609))
	model.add(Dense(1))
	adam = optimizers.Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='mean_squared_error', optimizer=adam)
	for i in range(nb_epoch):
		print('Epoch:',i)
		history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		loss_list.append(history.history['loss'][0])
		with open('./result/soh_loss.txt', 'a', encoding='utf-8') as f:
			f.write(str(history.history['loss'][0]) + "\n")
		model.reset_states()
		plot_model(model, to_file=r'./result/soh_model_structure.png', show_shapes=True)
	model.save(r'./result/soh_model.h5')
	return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 660, 4)
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(0, len(dataset)-look_back, 2641):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	dataY= np.array(dataY)        
	dataY = np.reshape(dataY,(dataY.shape[0],1))
	for i in range(len(dataY)):
		if dataY[i].astype("float64") == 0:
			dataY[i] = str(dataY[i-1][0].astype("float64"))
	dataset = np.concatenate((dataX,dataY),axis=1)
	return dataset,dataY


def experiment(series5, series6, series7, series18, series45, series46, series47, series48, series53, series54,
			   series55, series56, updates, look_back, neurons, n_epoch, batch_size):
	index = []
	raw_values5 = series5.values
	raw_values6 = series6.values
	raw_values7 = series7.values
	raw_values18 = series18.values
	raw_values45 = series45.values
	raw_values46 = series46.values
	raw_values47 = series47.values
	raw_values48 = series48.values
	raw_values53 = series53.values
	raw_values54 = series54.values
	raw_values55 = series55.values
	raw_values56 = series56.values
	raw_values = np.concatenate((raw_values5, raw_values6, raw_values7, raw_values18, raw_values45, raw_values46, raw_values47, raw_values48, raw_values53, raw_values54, raw_values55, raw_values56), axis=0)

	dataset, dataY = create_dataset(raw_values,look_back)
	dataset_5, dataY_5 = create_dataset(raw_values5,look_back)
	dataset_6, dataY_6 = create_dataset(raw_values6,look_back)
	dataset_7, dataY_7 = create_dataset(raw_values7,look_back)
	dataset_18, dataY_18 = create_dataset(raw_values18,look_back)
	dataset_45, dataY_45 = create_dataset(raw_values45,look_back)
	dataset_46, dataY_46 = create_dataset(raw_values46,look_back)
	dataset_47, dataY_47 = create_dataset(raw_values47,look_back)
	dataset_48, dataY_48 = create_dataset(raw_values48,look_back)
	dataset_53, dataY_53 = create_dataset(raw_values53,look_back)
	dataset_54, dataY_54 = create_dataset(raw_values54,look_back)
	dataset_55, dataY_55 = create_dataset(raw_values55,look_back)
	dataset_56, dataY_56 = create_dataset(raw_values56,look_back)

	train_size_5 = int(dataset_5.shape[0] * 0.7)
	train_size_6 = int(dataset_6.shape[0] * 0.7)
	train_size_7 = int(dataset_7.shape[0] * 0.7)
	train_size_18 = int(dataset_18.shape[0] * 0.7)
	train_size_45 = int(dataset_45.shape[0] * 0.7)
	train_size_46 = int(dataset_46.shape[0] * 0.7)
	train_size_47 = int(dataset_47.shape[0] * 0.7)
	train_size_48 = int(dataset_48.shape[0] * 0.7)
	train_size_53 = int(dataset_53.shape[0] * 0.7)
	train_size_54 = int(dataset_54.shape[0] * 0.7)
	train_size_55 = int(dataset_55.shape[0] * 0.7)
	train_size_56 = int(dataset_56.shape[0] * 0.7)

	# split into train and test sets
	train_5, test_5 = dataset_5[0:train_size_5], dataset_5[train_size_5:]
	train_6, test_6 = dataset_6[0:train_size_6], dataset_6[train_size_6:]
	train_7, test_7 = dataset_7[0:train_size_7], dataset_7[train_size_7:]
	train_18, test_18 = dataset_18[0:train_size_18], dataset_18[train_size_18:]
	train_45, test_45 = dataset_45[0:train_size_45], dataset_45[train_size_45:]
	train_46, test_46 = dataset_46[0:train_size_46], dataset_46[train_size_46:]
	train_47, test_47 = dataset_47[0:train_size_47], dataset_47[train_size_47:]
	train_48, test_48 = dataset_48[0:train_size_48], dataset_48[train_size_48:]
	train_53, test_53 = dataset_53[0:train_size_53], dataset_53[train_size_53:]
	train_54, test_54 = dataset_54[0:train_size_54], dataset_54[train_size_54:]
	train_55, test_55 = dataset_55[0:train_size_55], dataset_55[train_size_55:]
	train_56, test_56 = dataset_56[0:train_size_56], dataset_56[train_size_56:]

	train = np.concatenate((train_5, train_6, train_7, train_18, train_45, train_46, train_47, train_48, train_53, train_54, train_55, train_56), axis=0)
	np.random.shuffle(train)
	features = train[:, :-1]
	labels = train[:, -1]

	scaler, train_scaled, test5_scaled = scale(train, test_5)
	joblib.dump(scaler, r'.\result\scaler_soh.pickle')

	starttime = time.time()
	# fit the model
	lstm_model = fit_lstm(train_scaled, batch_size, n_epoch, neurons)
	endtime = time.time()
	dtime = endtime - starttime

	# forecast the entire training dataset to build up state for forecasting
	print('Forecasting Training Data')   
	predictions_train = list()
	for i in range(len(train_scaled)):
		# make one-step forecast
		X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, batch_size, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# store forecast
		predictions_train.append(yhat)
		expected = labels[i]
		print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, float(expected)))

	# report performance
	rmse_train = sqrt(mean_squared_error(np.array(labels).astype("float64")/2, np.array(predictions_train)/2))
	print('Train RMSE: %.3f' % rmse_train)
	index.append(rmse_train)

	# forecast the test data(#5)
	print('Forecasting Testing Data')
	predictions_test = list()
	for i in range(len(test5_scaled)):
		# make one-step forecast
		X, y = test5_scaled[i, 0:-1], test5_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, batch_size, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# store forecast
		predictions_test.append(yhat)
		expected = dataY_5[len(train_5) + i]
		print('Cycle=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

	# report performance using RMSE
	rmse_test = sqrt(mean_squared_error(dataY_5[-len(test5_scaled):].astype("float64")/2, np.array(predictions_test)/2))
	print('Test RMSE: %.3f' % rmse_test)
	print("程序训练时间：%.8s s" % dtime)

	index.append(rmse_test)
	index.append(dtime)
	with open(r'./result/soh_prediction_result.txt', 'a', encoding='utf-8') as f:
		for j in range(len(index)):
			f.write(str(index[j]) + "\n")

	with open(r'./result/soh_prediction_data_#5.txt', 'a', encoding='utf-8') as f:
		for k in range(len(predictions_test)):
			f.write(str(predictions_test[k]) + "\n")
		dataY_5 = np.array(dataY_5)
	# line plot of observed vs predicted
	fig, ax = plt.subplots(1)
	ax.plot(dataY_5[-len(test5_scaled):].astype("float64"), label='original', color='blue')
	ax.plot(predictions_test, label='predictions', color='red')
	ax.legend(loc='upper right')
	ax.set_xlabel("Cycle",fontsize = 16)
	ax.set_ylabel('Capacity '+ r'$(AH)$',fontsize = 16)
	plt.savefig(r'./result/soh_result.png')
	plt.show()


def run():
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

	look_back = 2640
	neurons = [64, 64]
	n_epochs = 153
	updates = 1
	batch_size = 14
	experiment(series1, series2, series3, series4, series5, series6, series7, series8, series9, series10,
			   series11, series12, updates,look_back, neurons, n_epochs, batch_size)


run()
fig = plt.figure()
plt.plot(loss_list, label='loss', color='blue')
plt.legend()
plt.title('model loss')
plt.savefig('./result/soh_loss.png')
plt.show()

