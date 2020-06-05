
import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_line(df, column_x, xlabel, ylabel, title, filename):

	plt.figure(figsize = (10,7))
	plt.title(title)
	plt.ylim(0.5, 1.01)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.errorbar(df[column_x], df['train_accuracy_mean'], df['train_accuracy_std'], linestyle='None', marker='^', color='red', label='Treinamento')
	plt.errorbar(df[column_x], df['validation_accuracy_mean'], df['validation_accuracy_std'], linestyle='None', marker='^', color='green', label='Validaao')
	plt.errorbar(df[column_x], df['test_accuracy_mean'], df['test_accuracy_std'], linestyle='None', marker='^', color='blue', label='Teste')

	plt.legend(loc='lower left')

	plt.savefig('results//graphs/' + filename + '.jpg')



def main():
	df = pd.read_csv('results/accuracy.csv')

	thresholds = df['threshold'].unique()

	for threshold in thresholds:

			print("Generating graph for threshold " + str(threshold))

			df_threshold = df[df['threshold'] == threshold]
			column_x = 'addressSize'
			xlabel = 'Numero de bits de enderecamento'
			ylabel = 'Acuracia'
			title = 'Resultados variando o numero de bits de enderecamento e threshold=' + str(threshold)
			filename = 'accuracy_addressSize_threshold_' + str(threshold)
			plot_line(df_threshold, column_x, xlabel, ylabel, title, filename)

if __name__ == "__main__":
	sys.exit(main())