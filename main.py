# -*- coding: utf-8 -*-

# Created by Thais Luca
# Systems Engineering and Computer Science Program - COPPE, Federal University of Rio de Janeiro
# Created at 05/01/2020
# Last update at 05/01/2020


import wisardpkg as wp
import pandas as pd
import numpy as np
import sys

from collections import Counter
from pandas import compat
compat.PY3 = True

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

def load_dataset():
	dataset = pd.read_csv('dataset/mnist_train.csv')
	dataset = pd.concat([dataset, pd.read_csv('dataset/mnist_test.csv')])
	return dataset

def preprocess(df):
	threshold = 125
	columns = df.columns
	for column in columns:
		df[column] = np.where(df[column] >= threshold, 1, 0)
	return df


def main():

	confusion_matrix_train_scores = []
	confusion_matrix_validation_scores = []
	confusion_matrix_test_scores = []

	addressSize = 3     # number of addressing bits in the ram
	ignoreZero  = False # optional; causes the rams to ignore the address 0

	# False by default for performance reasons,
	# when True, WiSARD prints the progress of train() and classify()
	verbose = False

	dataset = load_dataset()

	train, test = train_test_split(dataset, test_size=0.3)
	X = preprocess(train.drop(['label'], axis=1)).values.tolist()
	Y = train['label'].values.tolist()

	X_test = preprocess(test.drop(['label'], axis=1)).values.tolist()
	Y_test = test['label'].values.tolist()

	# Define model
	wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

	kf = KFold(n_splits=2)
	for train_index, val_index in kf.split(X):
		print("TRAIN:", len(train_index), "VALIDATION:", len(val_index))
		x_train, x_val = [X[index] for index in train_index], [X[index] for index in val_index]
		y_train, y_val = [str(Y[index]) for index in train_index], [str(Y[index]) for index in val_index]

		# train using the input data
		wsd.train(x_train,y_train)

		# classify train data
		out_train = wsd.classify(x_train)

		# classify validation data
		out_val = wsd.classify(x_val)

		confusion_matrix_train_scores.append(confusion_matrix(y_train, out_train))
		confusion_matrix_validation_scores.append(confusion_matrix(y_val, out_val))

		#classify test data
		out_test = wsd.classify(X_test)

		confusion_matrix_test_scores.append(confusion_matrix(Y_test, out_test))

	mean_of_conf_matrix_train_arrays = np.mean(confusion_matrix_train_scores, axis=0)
	print(mean_of_conf_matrix_train_arrays)


if __name__ == "__main__":
	sys.exit(main())