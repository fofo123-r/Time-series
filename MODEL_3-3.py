###################### IMPORTS ######################
import math

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)


###################### MODEL

def build_model(data):
	model = Sequential()
	model.add(LSTM(60, return_sequences=False, input_shape=(data.shape[1], 1)))
	model.add(Dense(21))
	model.add(Dense(1))
	return model


# Model: "sequential_4"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm_4 (LSTM)                (None, 60)                14880
# _________________________________________________________________
# dense_8 (Dense)              (None, 21)                1281
# _________________________________________________________________
# dense_9 (Dense)              (None, 1)                 22
# =================================================================
# Total params: 16,183
# Trainable params: 16,183
# Non-trainable params: 0
# _________________________________________________________________
# None




###################### MAIN


## Load Data
df = pd.read_csv('Stocks Data.csv', header = 1)
df = df.iloc[1:, :].rename(columns={'Symbols': 'Date'}).reset_index().drop('index',axis=1)
df.Date = pd.to_datetime(df.Date)
df_3yrs = df[df.Date >= '2018-01-02']

# Separate dataframe by stock
agys = df_3yrs[['Date', 'AGYS']]
mrin = df_3yrs[['Date', 'MRIN']].dropna()
mxl = df_3yrs[['Date', 'MXL']].dropna()

dell = pd.read_csv('DELL.csv')[['Date', 'Close']].rename(columns={'Close':'DELL'})
aapl = pd.read_csv('AAPL.csv')[['Date', 'Close']].rename(columns={'Close':'AAPL'})
dell = dell[dell.Date >= '2018-01-02']
aapl = aapl[aapl.Date >= '2018-01-02']

stocks = [aapl, agys, dell, mrin, mxl]
test_rmses = {}

for stock in stocks:
	batch_size = 5
	n_epochs = 120
	seq_len = 21
	## Normalize
	dataset = stock[stock.columns[1]].values
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(dataset.reshape(-1,1))
	scaled_data

	##. Split Data - Train, Test (80/20)
	train_size = int(len(stock) * 0.8)
	test_size = len(stock) - train_size
	train, test = stock[:train_size], stock[train_size:]

	train_size = int(len(scaled_data) * 0.8)
	test_size = len(scaled_data) - train_size
	train_scaled, test_scaled = scaled_data[:train_size,:], scaled_data[train_size:,:]


	## Preprocess data for one-step
	# Prep train
	x_train = np.empty((len(train_scaled) - seq_len, seq_len, 1))
	y_train = np.empty((len(train_scaled) - seq_len, 1))
	for i in range(len(train_scaled) - seq_len):
		x_train[i,:,:] = train_scaled[i:seq_len+i].reshape(-1,1)
		y_train[i,:] = train_scaled[seq_len+i].reshape(-1,1)

	# Prep test
	x_test = np.empty((len(test_scaled)-seq_len, seq_len, 1))
	y_test = np.empty((len(test_scaled)-seq_len, 1))
	for i in range(len(test_scaled)-seq_len):
		x_test[i,:,:] = test_scaled[i:seq_len+i].reshape(-1,1)
		y_test[i,:] = test_scaled[seq_len+i].reshape(-1,1)


	## Create model, compile, train
	model = build_model(x_train)
	model.compile(loss='mse', optimizer='adam')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(x_test, y_test))

	## Use model to predict on train and test x
	train_prediction = model.predict(x_train)
	test_prediction = model.predict(x_test)

	## Invert predictions
	# invert predictions
	train_prediction = scaler.inverse_transform(train_prediction)
	y_train = scaler.inverse_transform(y_train)
	test_prediction = scaler.inverse_transform(test_prediction)
	y_test = scaler.inverse_transform(y_test)


	## Calculate Metrics
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(y_train, train_prediction))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(y_test, test_prediction))
	print('Test Score: %.2f RMSE' % (testScore))
	test_rmses[stock.columns[1]] = testScore

	## Plot
	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(scaled_data)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[seq_len:len(train_prediction)+seq_len, :] = train_prediction
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(scaled_data)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(train_prediction)+(seq_len*2):len(scaled_data), :] = test_prediction
	# plot baseline and predictions
	plt.figure()
	plt.plot(scaler.inverse_transform(scaled_data), label='Actual')
	plt.plot(trainPredictPlot, label='Train Prediction')
	plt.plot(testPredictPlot, label='Test Prediction')
	plt.title('{} Stock Price Predictions'.format(stock.columns[1]))
	plt.xlabel('Time')
	plt.ylabel('Price')
	plt.legend()
	plt.show()

print(test_rmses)
print(model.summary())



