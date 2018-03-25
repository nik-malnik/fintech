import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, ensemble
import re

def create_dataset(ticker = 'YHOO', look_back=1):
	prices = pd.read_csv('prices.csv')
	dataset = prices[prices['symbol'] == ticker]
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = []
		for col in ['open','high','low','close']:
			a.append(list(dataset[col][i:(i+look_back)])[0])
		dataX.append(a)
		dataY.append(list(dataset['open'])[i + look_back])
	return np.around(np.array(dataX),decimals=3), np.around(np.array(dataY),decimals=3)

def combine_history(dataX,dataY,labels,look_back = 10):
	data = np.zeros((dataX.shape[0] - look_back + 1, dataX.shape[1]*look_back))
	for i in range(look_back-1,dataX.shape[0]):
		for j in range(look_back):
			data[i - look_back,j*dataX.shape[1]:(j+1)*dataX.shape[1]] = dataX[i - look_back + j,:]
	
	return data, dataY[look_back - 1:], labels[look_back-1:]

def svm_hyper_parameter_tuning(dataX,labels, train_size):
	train_perf = []
	test_perf = []
	for i in range(20):
		C = 0.01*np.power(2,i)
		model_svm = svm.SVC(C = C).fit(dataX[:train_size],labels[:train_size])
		predictions = ( model_svm.predict(dataX[:train_size]) > 0 )
		train_perf.append(np.sum( predictions == labels[:train_size])/(1.0*len(labels[:train_size])))	
		predictions = ( model_svm.predict(dataX[train_size:]) > 0 )
		test_perf.append(np.sum( predictions == labels[train_size:])/(1.0*len(labels[train_size:])))
		
	plt.plot(range(20),train_perf)
	plt.plot(range(20),test_perf)
	plt.show()

def clean_headline(headline):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", headline).split())
