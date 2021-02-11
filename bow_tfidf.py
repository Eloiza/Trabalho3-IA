# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier

def main():
	dataset = pd.read_csv('datasets/imdb.csv')
	dataset.head()
	dataset = dataset[:1000]

	count = CountVectorizer(stop_words='english')
	X = count.fit_transform(dataset['review'])

	print("X.shape: ", X.shape)

	tfidf = TfidfTransformer()
	X = tfidf.fit_transform(X)

	print("X.shape: ", X.shape)

	X_train, X_test, y_train, y_test = train_test_split(X, dataset['sentiment'], test_size=0.33)

	print("X_train.shape: ", X_train.shape)
	print("X_test.shape: ", X_test.shape)

	model = MLPClassifier(max_iter=300)
	model.fit(X_train, y_train)
	print("Acurácia MLP", model.score(X_test, y_test))

if __name__ == '__main__':
	main()