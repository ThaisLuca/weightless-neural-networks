# -*- coding: utf-8 -*-

# Created by Thais Luca
# Systems Engineering and Computer Science Program - COPPE, Federal University of Rio de Janeiro
# Created at 05/01/2020
# Last update at 05/01/2020


import wisardpkg as wp
import pandas as pd
import numpy as np
import cv2
import sys

from pandas import compat
compat.PY3 = True

from sklearn.model_selection import train_test_split

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

	dataset = load_dataset()

	train, test = train_test_split(dataset, test_size=0.3)
	x_train = preprocess(train.drop(['label'], axis=1)).values.tolist()
	y_train = train['label'].values.tolist()
	x_test = preprocess(test.drop(['label'], axis=1)).values.tolist()
	y_test = test['label'].values.tolist()

	addressSize = 3     # number of addressing bits in the ram
	ignoreZero  = False # optional; causes the rams to ignore the address 0

	# False by default for performance reasons,
	# when True, WiSARD prints the progress of train() and classify()
	verbose = True

	wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

	# train using the input data
	wsd.train(x_train,y_train)

	# classify some data
	out = wsd.classify(x_train)

	# the output of classify is a string list in the same sequence as the input
	for i,d in enumerate(x_train):
	    print(out[i],d)

if __name__ == "__main__":
	sys.exit(main())