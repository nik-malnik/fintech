import numpy as np	# for basic algebra
import matplotlib.pyplot as plt   # for Plots
import price_prediction_utils as ppu	# Helper Functions to fetch data
from sklearn import linear_model, svm, ensemble	# Machine Learning Algorithms

''' Keras for building Neural Networks '''
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


''' Fetch data '''
dataX, dataY = ppu.create_dataset(ticker = 'YHOO', look_back=1)
labels = (dataY - dataX[:,0]) > 0.0
#pd.DataFrame({'open': dataX[:,0], 'high': dataX[:,1], 'low': dataX[:,2], 'close': dataX[:,3], 'y': dataY, 'label':1.0*labels}).to_csv('yahoo_prices.csv')

''' Split train vs test '''
train_size = int(len(dataX) * 0.80)

features = [0,1,2,3]

''' OLS '''
model_ols = linear_model.LinearRegression().fit(dataX[:train_size,features],labels[:train_size])
predictions = ( model_ols.predict(dataX[train_size:,features]) > 0.5 )
print "OLS Accuracy:" + str( np.mean( predictions == labels[train_size:]) )


''' Logistic Regression '''
model_logit = linear_model.LogisticRegression().fit(dataX[:train_size,features],labels[:train_size])
predictions = ( model_logit.predict(dataX[train_size:,features]) > 0 )
print "Logit Accuracy:" + str( np.mean( predictions == labels[train_size:]))

''' Random Forest '''
model_forest = ensemble.RandomForestClassifier().fit(dataX[:train_size,features],labels[:train_size])
predictions = ( model_forest.predict(dataX[train_size:,features]) > 0 )
print "Random Forest Accuracy:" + str( np.mean( predictions == labels[train_size:]) )

plt.scatter(range(len(dataY)), dataY, c = (labels == ( model_forest.predict(dataX[:,features]) > 0 )))


''' Support Vector Machine '''
#ppu.svm_hyper_parameter_tuning(dataX,labels, train_size)
model_svm = svm.SVC(C = 10.0).fit(dataX[:train_size,features],labels[:train_size])
predictions = ( model_svm.predict(dataX[train_size:,features]) > 0 )
print "SVM Accuracy:" + str( np.mean( predictions == labels[train_size:]) )
	

''' Neural Network - Basic '''

def NN_Basic(dataX,dataY,labels,look_back = 10):
	
	dataX,dataY,labels = ppu.combine_history(dataX,dataY,labels,look_back = look_back)
	dataX = (dataX - np.min(dataX)) / (np.max(dataX) - np.min(dataX))
	input_dim = dataX.shape[1]
	
	model = Sequential()
	model.add(Dense(64, activation='relu', input_dim=input_dim))
	model.add(Dropout(0.2))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
	
	history = model.fit(dataX, labels*1.0, epochs=1000, batch_size=8)
	print "Neural Network Basic Accuracy:" + str(model.test_on_batch(dataX,labels)[1])
	plt.plot(range(1000),history.history['acc'])
	plt.show()
	
	return model

model = NN_Basic(dataX[:,features],dataY,labels,look_back = 10)
	
''' Neural Network - LSTM '''

def NN_LSTM(dataX,dataY,labels, look_back):
	
	trainX = np.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))

	model = Sequential()
	model.add(LSTM(input_dim=4,output_dim=16))
	model.add(Dropout(0.2))
	model.add(Dense(1,activation='sigmoid'))

	model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
	
	history = model.fit(trainX,labels,epochs=1000,batch_size=look_back)
	print "Neural Network LSTM Accuracy:" + str(model.test_on_batch(trainX,labels)[1])
	plt.plot(range(1000),history.history['acc'])
	plt.show()

model = NN_LSTM(dataX,dataY,labels,look_back = 16)

