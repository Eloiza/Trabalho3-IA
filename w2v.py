# -*- coding: utf-8 -*-
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import nltk
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def main():
	dataset = pd.read_csv('datasets/imdb.csv')
	print(dataset.head())
	dataset = dataset[:1000]

	X = [remove_stopwords(sent) for sent in dataset['review']]
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

	print("Acurácia MLP", mlp.score(X_test_v, y_test))

if __name__ == '__main__':
	main()