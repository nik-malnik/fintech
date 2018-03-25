import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt  
import price_prediction_utils as ppu
from sklearn import feature_extraction
from sklearn import linear_model, svm, ensemble

''' Keras for building Neural Networks '''
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
    
df_news = pd.read_csv('Combined_News_DJIA.csv')
N = 1	# Number of news headlines to consider
for i in range(1,N):
	df_news['Top' + str(i) ] = map(lambda x: ppu.clean_headline(x),df_news['Top' + str(i)])

train_size = int(len(dataX) * 0.80)

model_text = feature_extraction.text.CountVectorizer(max_features=100).fit(df_news['Top1'])
dataX = model_text.transform(df_news['Top1']).todense()
labels = np.array(df_news['Label'])

''' OLS '''
model_ols = linear_model.LinearRegression().fit(dataX,labels)
predictions = ( model_ols.predict(dataX) > 0.5 )
print "OLS Accuracy:" + str( np.mean( predictions == labels) )

''' Logistic Regression '''
model_logit = linear_model.LogisticRegression().fit(dataX,labels)
predictions = ( model_logit.predict(dataX) > 0 )
print "Logit Accuracy:" + str( np.mean( predictions == labels))

''' Random Forest '''
model_forest = ensemble.RandomForestClassifier(min_samples_leaf=5).fit(dataX,labels)	
predictions = ( model_forest.predict(dataX) > 0 )
print "Random Forest Accuracy:" + str( np.mean( predictions == labels) )

''' Support Vector Machine '''
#ppu.svm_hyper_parameter_tuning(dataX,labels, train_size)
model_svm = svm.SVC(C = 12.5).fit(dataX,labels)
predictions = ( model_svm.predict(dataX) > 0 )
print "SVM Accuracy:" + str( np.mean( predictions == labels) )

''' Neural Network - Basic '''

def NN_Basic(dataX,labels):
	
	input_dim = dataX.shape[1]
	
	model = Sequential()
	model.add(Dense(64, activation='relu', input_dim=input_dim))
	model.add(Dropout(0.5))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
	
	history = model.fit(dataX, labels*1.0, epochs=1000, batch_size=64)
	print "Neural Network Basic Accuracy:" + str(model.test_on_batch(dataX,labels)[1])
	print history.history.keys()
	plt.plot(range(1000),history.history['acc'])
	plt.show()
	
	return model

model = NN_Basic(dataX, labels)
