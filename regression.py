from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import StratifiedKFold, KFold
import sklearn.metrics as metrics
from math import sqrt

## For reproducibility
from numpy.random import seed
seed(10241996)
from tensorflow import set_random_seed
set_random_seed(10241996)


class Dataset():
	def __init__(self, filename):
		self.file = filename
		self.x_data = None
		self.y_data = None
		self.y_scaler = MinMaxScaler()
		self.x_scaler = MinMaxScaler()


	def loadDataset(self,scaled=True):
		## Reading .csv file using pandas
		df = pd.read_csv(self.file)
		labels = ["Appliances", "Lights"]
		features = [i for i in df.columns if i not in labels]
		
		## Separate output and input
		self.y_data = df[labels]
		self.x_data = df[features]

		## Normalize output table
		if scaled:
			self.y_data = self.y_scaler.fit_transform(self.y_data)
			self.x_data = self.x_scaler.fit_transform(self.x_data)
		else:
			self.y_data = self.y_data.as_matrix()
			self.x_data = self.x_data.as_matrix()

	def kfoldDataset(self, scaled=True):
		df = pd.read_csv(self.file)
		labels = ["Appliances", "Lights"]
		features = [i for i in df.columns if i not in labels]
		
		## Separate output and input
		self.y_data = df[labels]
		self.x_data = df[features]

		## Normalize output table
		if scaled:
			self.y_data = self.y_scaler.fit_transform(self.y_data)
			self.x_data = self.x_scaler.fit_transform(self.x_data)
		else:
			self.y_data = self.y_data.as_matrix()
			self.x_data = self.x_data.as_matrix()

	def transformData(self, array, isLabel=True):
		if isLabel:
			return(self.y_scaler.inverse_transform(array))
		else:
			return(self.x_scaler.inverse_transform(array))


def train(data, layers={1:[100,"relu"]},epochs=100, kfold=2, metrics=["mse","mae"],issave=True, savePath=os.path.dirname(os.path.abspath(__file__)),filename="model"):
	## Training of the x_ and y_ dataset using Keras
	## Save trained model to savePath
	cvscores_train=dict()
	cvscores_test = dict()
	for m in metrics:
		cvscores_train[m] = 0
		cvscores_test[m] = 0

	## Create model
	model = Sequential()
		
	for layer in layers.keys():
		if layer==1:
			neurons = layers.get(layer)[0]
			activation = layers.get(layer)[1]
			model.add(Dense(neurons, input_dim=len(data.x_data[0]), activation=activation))

		elif layer == max(layers.keys()):
			activation = layers.get(layer)[0]
			model.add(Dense(len(data.y_data[0]), activation=activation))

		else:
			neurons = layers.get(layer)[0]
			activation = layers.get(layer)[1]
			model.add(Dense(neurons, activation=activation))

	model.compile(loss="mse", optimizer="adam",metrics=metrics)
	model.summary()


	## Perform crossfold validation
	folds = KFold(n_splits=kfold, shuffle=True, random_state=102442357)
	fold=1
	for train, test in folds.split(data.x_data, data.y_data):

		model.fit(data.x_data[train],data.y_data[train],epochs=epochs, verbose=1)
		
		## Evaluate the model
		train_scores = model.evaluate(data.x_data[train], data.y_data[train], verbose=0)
		test_scores = model.evaluate(data.x_data[test], data.y_data[test], verbose=0)
		for m in cvscores_train.keys():
	
			cvscores_train[m]+=train_scores[metrics.index(m)+1]
			cvscores_test[m]+=test_scores[metrics.index(m)+1]
			# print(m, metrics.index(m),cvscores_train[m],cvscores_test[m])
		fold+=1

	print("\n================  %d-FOLD CV SUMMARY  ==================" %(kfold))
	for m in metrics:
		print("  [***]Training %s: %.5f\n" % (m, (cvscores_train[m]/kfold)))
	
	for m in metrics:
		print("  [***]Testing %s: %.5f\n" % (m, (cvscores_test[m]/kfold)))
	print("==========================================================")

	### SAVE MODEL HERE TO SAVEPATH
	if issave:
		## SAving the model
		model_json = model.to_json()
		with open(filename+".json", "w") as json_file:
		    json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(filename+".h5")
		# print("Saved model to disk")
		print("\nSaving '%s'(.json/.h5) to %s" %(filename,savePath))

	print("\nDone training...")


def modelReloaded(filename,savePath=os.path.dirname(os.path.abspath(__file__))):
	## create new model prediction using trained weights
	json_file = open(savePath+'/%s.json'%(filename), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(savePath+'/%s.h5'%(filename))
	print("Loaded '%s' from %s" %(filename,savePath))

	
	## Predicts and stores in an array
	return loaded_model
	

class computeMetrics():
	## Class of metric valuation

	def __init__(self, model):
		self.y_pred = None
		self.y_test = None
		self.model = model

	def denormalize(self, scaler):
		self.y_test = scaler.inverse_transform(self.y_test)
		self.y_pred = scaler.inverse_transform(self.y_pred)

	def predict(self,x_test):
		self.y_pred = self.model.predict(x_test)

	def rmse(self,x_test,y_test, scaler=None):
		## Compute for the RMSE
		self.y_test = y_test
		self.predict(x_test)
		if scaler is not None:
			self.denormalize(scaler)

		## equation for RMSE
		rms = sqrt(metrics.mean_squared_error(self.y_test, self.y_pred))
		return(rms)

	def rsquared(self,x_test,y_test,scaler=None):
		self.y_test = y_test
		self.predict(x_test)
		if scaler is not None:
			self.denormalize(scaler)

		## Compute for the R^2
		r2 = metrics.r2_score(self.y_test, self.y_pred)
		return(r2)


def main():
	## Setup directory. y default, script directory will be used
	savePath = os.path.dirname(os.path.abspath(__file__))
	filename = "modelAll"
	## Load dataset
	print("Loading training and validation dataset..")
	data = Dataset(sys.argv[1])
	data.kfoldDataset()
	print("Loaded. Start training...")

	## Create model
	## Save model
	layers={1: [100,"relu"],
			2: [100,"relu"],
			3: [100,"relu"],
			4: ["relu"]}

	## Training
	train(data, layers=layers,issave=1,epochs = 150, filename=filename)

	## Loading for prediction
	print("Loading testing dataset...")
	data_test = Dataset(sys.argv[2])
	data_test.loadDataset(scaled=False)

	model = modelReloaded(filename)
	prediction = model.predict(data_test.x_data)
	true = data_test.y_data
	for i in range(len(prediction)):
		print("True: ",true[i], "Pred: ",prediction[i])




if __name__ == "__main__":
	main()
	print("done..")