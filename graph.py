
import pandas as pd
import matplotlib.pyplot as plt

def plot_line(column_x, xlabel, ylabel, title, filename):

	df = pd.read_csv('results/accuracy.csv')

	plt.figure(figsize = (10,7))
	plt.title(title)
	plt.ylim(0.2, 1.01)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.errorbar(df[column_x], df['train_accuracy_mean'], df['train_accuracy_std'], linestyle='None', marker='^', color='red', label='Treinamento')
	plt.errorbar(df[column_x], df['validation_accuracy_mean'], df['validation_accuracy_std'], linestyle='None', marker='^', color='green', label='Validaao')
	plt.errorbar(df[column_x], df['test_accuracy_mean'], df['test_accuracy_std'], linestyle='None', marker='^', color='blue', label='Teste')

	#plt.plot(df[column_x], df['train_accuracy'], color='red', label='Treinamento')
	#plt.plot(df[column_x], df['validation_accuracy'], color='blue', label='Validacao')
	#plt.plot(df[column_x], df['test_accuracy'], color='orange', label='Teste')

	plt.legend(loc='lower left')

	plt.savefig('results/' + filename + '.jpg')

plot_line('addressSize', 'Numero de bits de enderecamento', 'Acuracia', 'Resultados variando o numero de bits de enderecamento', 'accuracy_addressSize')
plot_line('threshold', 'Threshold', 'Acuracia', 'Resultados variando o threshold', 'accuracy_threshold')