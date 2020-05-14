# -*- coding: utf-8 -*-

# Created by Thais Luca
# Systems Engineering and Computer Science Program - COPPE, Federal University of Rio de Janeiro
# Created at 05/01/2020
# Last update at 05/01/2020


import wisardpkg as wp
import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt
import seaborn as sn

from collections import Counter
from pandas import compat
compat.PY3 = True
pd.options.display.float_format = '{:.2f}'.format

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

def plot_heatmap(cm, title, filename):
	df_cm = pd.DataFrame(cm, index = range(10), columns = range(10))
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, cmap="YlGnBu", annot=True)
	plt.title(title)
	plt.ylabel('Target Value')
	plt.xlabel('Predicted Value')
	plt.savefig('results/' + filename + '.pdf')
	#plt.show()

def main():

	confusion_matrix_train_scores = np.zeros((10,10))
	confusion_matrix_validation_scores = np.zeros((10,10))
	confusion_matrix_test_scores = np.zeros((10,10))

	addressSize = 3     # number of addressing bits in the ram
	ignoreZero  = False # optional; causes the rams to ignore the address 0
	n_splits = 10		# number of splits used in KFold

	# False by default for performance reasons,
	# when True, WiSARD prints the progress of train() and classify()
	verbose = False

	dataset = load_dataset()

	train, test = train_test_split(dataset, test_size=0.3)
	X = preprocess(train.drop(['label'], axis=1)).values.tolist()
	Y = train['label'].values.tolist()

	X_test = preprocess(test.drop(['label'], axis=1)).values.tolist()
	Y_test = test['label'].values.tolist()
	Y_test = [str(y) for y in Y_test]

	# Define model
	wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

	kf = KFold(n_splits=n_splits)
	fold = 1
	for train_index, val_index in kf.split(X):
		print("FOLD:", fold)
		x_train, x_val = [X[index] for index in train_index], [X[index] for index in val_index]
		y_train, y_val = [str(Y[index]) for index in train_index], [str(Y[index]) for index in val_index]

		#print("Training: ", Counter(y_train))
		#print("Validation: ", Counter(y_val))
		#print("Test: ", Counter(Y_test))

		# train using the input data
		print("Training")
		wsd.train(x_train,y_train)

		# classify train data
		print("Train data classification")
		out_train = wsd.classify(x_train)
		cm_training = confusion_matrix(y_train, out_train)
		cm_training = cm_training / cm_training.astype(np.float).sum(axis=1)
		confusion_matrix_train_scores += cm_training

		# classify validation data
		print("Validation data classification")
		out_val = wsd.classify(x_val)
		cm_validation = confusion_matrix(y_val, out_val)
		cm_validation = cm_validation / cm_validation.astype(np.float).sum(axis=1)
		confusion_matrix_validation_scores += cm_validation

		#classify test data
		print("Test data classification")
		out_test = wsd.classify(X_test)
		cm_test = confusion_matrix(Y_test, out_test)
		cm_test = cm_test / cm_test.astype(np.float).sum(axis=1)
		confusion_matrix_test_scores += cm_test
		
		fold += 1
		print('\n')

	confusion_matrix_train_scores = np.divide(confusion_matrix_train_scores, n_splits)
	confusion_matrix_validation_scores = np.divide(confusion_matrix_validation_scores, n_splits)
	confusion_matrix_test_scores = np.divide(confusion_matrix_test_scores, n_splits)

	plot_heatmap(confusion_matrix_train_scores, title='Training', filename='training_heatmap')
	plot_heatmap(confusion_matrix_validation_scores, title='Validation', filename='validation_heatmap')
	plot_heatmap(confusion_matrix_test_scores, title='Test', filename='test_heatmap')

if __name__ == "__main__":
	sys.exit(main())