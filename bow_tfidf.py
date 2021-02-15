# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from removeWords import VocabularyFilter
from prettyCMplot import plot_confusion_matrix

def main():
	#reads the dataset
	dataset_path = 'datasets/imdb.csv'
	print("Reading dataset file at ", dataset_path)
	dataset = pd.read_csv(dataset_path)
	# dataset = dataset[:100]

	X = dataset['review']

	# frequency = 0.8
	# print("Removing words that appear with %.2f frequency" %(frequency*100))
	# # removes words that appear >= 60% in the texts
	# X = VocabularyFilter().removeWords(X, frequency)

	count = CountVectorizer(stop_words='english')
	X = count.fit_transform(X)

	tfidf = TfidfTransformer()
	X = tfidf.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, dataset['sentiment'], test_size=0.33)

	print("Data ready for training...")
	print("Shape train set: ", X_train.shape)
	print("Shape test set : ", X_test.shape)

	print("Training MLP Classifier")
	model = MLPClassifier(max_iter=300)
	model.fit(X_train, y_train)

	print("Training done...Evaluating the model")
	y_pred = model.predict(X_test)

	y_test = list(y_test)

	loss_values = model.loss_curve_
	acc = accuracy_score(y_test, y_pred)
	f1Score = f1_score(y_test, y_pred, pos_label='negative')
	cm = confusion_matrix(y_test, y_pred)

	print("\nAcurácia MLP: %.2f" %(acc*100))
	print("F1 Score    : %.2f" %(f1Score*100))
	print("Loss 	   : %.4f" %(loss_values.pop()))
	print("Matriz de Confusão:")
	print(cm)

	# plt.ylabel('loss value')
	# plt.xlabel('epochs')
	# plt.title('MLP Trainig Loss Value')
	# plt.grid(True)
	# plt.plot(loss_values)
	# save_name = 'Results/bow_tfidf_' + str(int(frequency*100)) + '_loss' 
	# plt.savefig(save_name)

	# new_plt = plot_confusion_matrix(cm = cm, target_names = ['negative', 'positive'], title='MLP Confusion Matrix', cmap=None, normalize=True)
	# save_name = 'Results/bow_tfidf_' + str(int(frequency*100)) + '_confusion_matrix'
	# new_plt.savefig(save_name)

if __name__ == '__main__':
	main()