# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk

import matplotlib.pyplot as plt

from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from removeWords import VocabularyFilter
from prettyCMplot import plot_confusion_matrix

def main():
	dataset_path = 'datasets/imdb.csv'
	print("Reading dataset file at ", dataset_path)
	dataset = pd.read_csv('datasets/imdb.csv')
		
	X = dataset['review']
	#remove words that appear >= 0.6
	# frequency = 0.30
	# print("Removing words that appear with %.2f frequency" %(frequency*100))
	# X = VocabularyFilter().removeWords(X, frequency)

	X = [remove_stopwords(sent) for sent in X]
	X = [simple_preprocess(sent, deacc=True) for sent in X]

	X_train, X_test, y_train, y_test = train_test_split(X, dataset['sentiment'], test_size=0.33)

	model = Word2Vec(X, min_count=1, size=300, window=10)
	model.train(X, total_examples=len(X), epochs=30)

	X_train_v = []
	for sent in X_train:
	    sent_vector = np.mean([model.wv[word] for word in sent if word in model.wv], axis=0)
	    X_train_v.append(sent_vector)

	    
	X_test_v = []
	for sent in X_test:
	    sent_vector = np.mean([model.wv[word] for word in sent if word in model.wv], axis=0)
	    X_test_v.append(sent_vector)

	print("Data ready for training...")
	print("Shape train set: ", np.array(X_train_v).shape)
	print("Shape test set : ", np.array(X_test_v).shape)

	print("Trainig MLP Classifier")
	mlp = MLPClassifier(max_iter=300)
	mlp.fit(X_train_v, y_train)

	print("Trainig done... Evaluating the model")
	y_pred = mlp.predict(X_test_v)

	y_test = list(y_test)

	loss_values = mlp.loss_curve_
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
	# save_name = 'Results/w2v_' + str(int(frequency*100)) + '_loss' 
	# plt.savefig(save_name)

	# new_plt = plot_confusion_matrix(cm = cm, target_names = ['negative', 'positive'], title='MLP Confusion Matrix', cmap=None, normalize=True)
	# save_name = 'Results/w2v_' + str(int(frequency*100)) + '_confusion_matrix'
	# new_plt.savefig(save_name)

if __name__ == '__main__':
	main()