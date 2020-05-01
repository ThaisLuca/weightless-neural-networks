# -*- coding: utf-8 -*-

# Created by Thais Luca
# Systems Engineering and Computer Science Program - COPPE, Federal University of Rio de Janeiro
# Created at 05/01/2020
# Last update at 05/01/2020


import wisardpkg as wp
import pandas as pd
import numpy as np
import sys

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

	accuracy_train_scores = []

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

		print(confusion_matrix(y_train, out_train))

		return


if __name__ == "__main__":
	sys.exit(main())