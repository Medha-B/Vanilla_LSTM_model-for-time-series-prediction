import numpy as np
import pandas as pd
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 30, 10

file = pd.ExcelFile(r"C:\Users\file_name.xlsx")
#loading the dataset
dataset = file.parse('Sheet1')

# parse strings to datetime type
dataset['Date'] = pd.to_datetime(dataset['Date'], infer_datetime_format= True)
indexedDataset = dataset.set_index(['Date'])

from datetime import datetime
indexedDataset.head()

import math
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# it is a good idea to fix the random number seed to ensure our results are reproducible
np.random.seed(7)

#normalizing the dataset
scaler = MinMaxScaler(feature_range=(0,1))
indexedDataset = scaler.fit_transform(indexedDataset)

#splitting the dataset into train and test sets
train_size = int(len(indexedDataset)*0.67)
test_size = len(indexedDataset) - train_size
train, test = indexedDataset[0:train_size,:], indexedDataset[train_size:len(dataset),:]

len(train)
len(test)

#we need a function to convert the numpy array into a dataset matrix. function will take two arguments, indexedDataset and look_back. look_backis the number of previous time steps to use as input variables to predict the next time period — in this case defaulted to 1.
def create_dataset (indexedDataset, look_back=1):
    dataX, dataY = [],[]
    for i in range(len(indexedDataset)-look_back):
        a = indexedDataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(indexedDataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)

# data needs to be reshaped to the form x=t and y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#lstm expects the data to be of the form [samples, time steps, features]. currently data is in the form of [samples, features]. so, the data needs to be reshaped uch that it includes the required time step.
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#create and fit the LSTM network. network has 1 visible layer with 1 input , 4 hidden layers, and 1 output layer which gives a single valued prediction.
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs= 100, batch_size=1, verbose=2)

#make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#invert the predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

#calculate the root mean square error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(indexedDataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(indexedDataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)-1:len(indexedDataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(indexedDataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

plt.plot(trainPredict)
plt.plot(testPredict)

