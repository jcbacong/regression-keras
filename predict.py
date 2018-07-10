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




def main():
	# name = sys.argv[1]
	json_file = open("modelAll.json" , 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("modelAll.h5")
	print("Loaded model...")
	 
	# evaluate loaded model on test data
	df = pd.read_csv(sys.argv[1])
	labels = ["Appliances", "Lights"]
	features = [i for i in df.columns if i not in labels]
	
	## Separate output and input
	output_table = df[labels].as_matrix()
	input_table = df[features].as_matrix()

	predictions = loaded_model.predict(input_table)
	for i in range(len(output_table)):
		print(output_table[i], predictions[i])
	# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



if __name__ == "__main__":
	main()
	print("done")
