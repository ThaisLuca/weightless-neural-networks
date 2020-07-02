# -*- coding: utf-8 -*-

# Created by Thais Luca
# Systems Engineering and Computer Science Program - COPPE, Federal University of Rio de Janeiro
# Created in 05/01/2020
# Last update in 06/05/2020


import wisardpkg as wp
import pandas as pd
import numpy as np
import sys, os

import matplotlib.pyplot as plt
import seaborn as sn

from collections import Counter
from pandas import compat
compat.PY3 = True
pd.options.display.float_format = '{:.2f}'.format

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score

def load_dataset():
	dataset = pd.read_csv('dataset/mnist_train.csv')
	dataset = pd.concat([dataset, pd.read_csv('dataset/mnist_test.csv')])
	return dataset

def preprocess(df, threshold):
	columns = df.columns
	for column in columns:
		df[column] = np.where(df[column] >= threshold, 1, 0)
	return df

def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df

def save_accuracy(threshold, addressSize, train_accuracy_score, validation_accuracy_score, test_accuracy_score):

	train_accuracy_score_mean = np.mean(train_accuracy_score, axis=0)
	validation_accuracy_score_mean = np.mean(validation_accuracy_score, axis=0)
	test_accuracy_score_mean = np.mean(test_accuracy_score, axis=0)

	train_accuracy_score_std = np.std(train_accuracy_score)
	validation_accuracy_score_std = np.std(validation_accuracy_score)
	test_accuracy_score_std = np.std(test_accuracy_score)

	matrix = {}
	matrix['threshold'] = [threshold]
	matrix['addressSize'] = [addressSize]
	matrix['train_accuracy_mean'] = [train_accuracy_score_mean]
	matrix['validation_accuracy_mean'] = [validation_accuracy_score_mean]
	matrix['test_accuracy_mean'] = [test_accuracy_score_mean]
	matrix['train_accuracy_std'] = [train_accuracy_score_std]
	matrix['validation_accuracy_std'] = [validation_accuracy_score_std]
	matrix['test_accuracy_std'] = [test_accuracy_score_std]

	result = pd.DataFrame(matrix)
	with open('results/accuracy.csv', 'a') as file:
		result.to_csv(file, index=False, header=True)

def save_matrix(cm, filename):
	df = cm2df(cm, range(10))

	with open(filename, 'a') as file:
		df.to_csv(file)

def plot_heatmap(cm, title, filename):
	df_cm = pd.DataFrame(cm, index = range(10), columns = range(10))
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, cmap="YlGnBu", annot=True)
	plt.title(title)
	plt.ylabel('Target Value')
	plt.xlabel('Predicted Value')
	plt.savefig('results/' + filename + '.jpg')
	#plt.show()

def main(input_threshold, input_addressSize):

	confusion_matrix_train_scores = np.zeros((10,10))
	confusion_matrix_validation_scores = np.zeros((10,10))
	confusion_matrix_test_scores = np.zeros((10,10))

	train_accuracy_score = [] 
	validation_accuracy_score = []
	test_accuracy_score = []

	addressSize = input_addressSize     # number of addressing bits in the ram
	ignoreZero  = False 				# optional; causes the rams to ignore the address 0
	n_splits = 10						# number of splits used in KFold
	threshold = input_threshold

	# False by default for performance reasons,
	# when True, WiSARD prints the progress of train() and classify()
	verbose = True

	dataset = load_dataset()

	train, test = train_test_split(dataset, test_size=0.3)
	print(len(train), len(test))
	X = preprocess(train.drop(['label'], axis=1), threshold).values.tolist()
	Y = train['label'].values.tolist()

	X_test = preprocess(test.drop(['label'], axis=1), threshold).values.tolist()
	Y_test = test['label'].values.tolist()
	Y_test = [str(y) for y in Y_test]

	# Define model
	wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

	kf = KFold(n_splits=n_splits, shuffle=True)
	fold = 1
	for train_index, val_index in kf.split(X):
		print("FOLD:", fold)
		print("TRAIN: {} VALIDATION: {}".format(len(train_index), len(val_index)))
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
		train_accuracy_score.append(accuracy_score(y_train, out_train))


		# classify validation data
		print("Validation data classification")
		out_val = wsd.classify(x_val)

		cm_validation = confusion_matrix(y_val, out_val)
		cm_validation = cm_validation / cm_validation.astype(np.float).sum(axis=1)
		confusion_matrix_validation_scores += cm_validation
		validation_accuracy_score.append(accuracy_score(y_val, out_val))

		#classify test data
		print("Test data classification")
		out_test = wsd.classify(X_test)

		cm_test = confusion_matrix(Y_test, out_test)
		cm_test = cm_test / cm_test.astype(np.float).sum(axis=1)
		confusion_matrix_test_scores += cm_test
		test_accuracy_score.append(accuracy_score(Y_test, out_test))
		
		fold += 1
		print('\n')

	confusion_matrix_train_scores = np.divide(confusion_matrix_train_scores, n_splits)
	confusion_matrix_validation_scores = np.divide(confusion_matrix_validation_scores, n_splits)
	confusion_matrix_test_scores = np.divide(confusion_matrix_test_scores, n_splits)

	save_accuracy(threshold, addressSize, train_accuracy_score, validation_accuracy_score, test_accuracy_score)

	path = 'results/threshold_' + str(threshold) + 'addressSize_' + str(addressSize)
	if(not os.path.isdir(path)):
		os.mkdir(path)

	plot_heatmap(confusion_matrix_train_scores, title='Training', filename=path + '/training_heatmap')
	plot_heatmap(confusion_matrix_validation_scores, title='Validation', filename=path + '/validation_heatmap')
	plot_heatmap(confusion_matrix_test_scores, title='Test', filename=path + '/test_heatmap')

	save_matrix(confusion_matrix_train_scores, path + '/training_confusion_matrix.csv')
	save_matrix(confusion_matrix_validation_scores,  path + '/validation_confusion_matrix.csv')
	save_matrix(confusion_matrix_test_scores,  path + '/test_confusion_matrix.csv')
