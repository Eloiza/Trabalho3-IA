# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk

from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from removeWords import VocabularyFilter

def main():
	dataset = pd.read_csv('datasets/imdb.csv')
	# dataset = dataset[:100]
		
	#remove words that appear >= 0.6
	X = dataset['review']
	X = VocabularyFilter().removeWords(X, 0.1)

	X = [remove_stopwords(sent) for sent in X]
	X = [simple_preprocess(sent, deacc=True) for sent in X]

	print("len(X)", len(X))
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

	mlp = MLPClassifier(max_iter=300)
	mlp.fit(X_train_v, y_train)

	y_pred = mlp.predict(X_test_v)

	y_test = list(y_test)

	loss_values = mlp.loss_curve_
	acc = accuracy_score(y_test, y_pred)
	f1Score = f1_score(y_test, y_pred, pos_label='negative')
	cm = confusion_matrix(y_test, y_pred)


	print("\nAcurácia MLP: %.2f" %(acc*100))
	print("F1 Score    : %.2f" %(f1Score*100))
	print("Matriz de Confusão:")
	print(cm)


if __name__ == '__main__':
	main()