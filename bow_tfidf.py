# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from removeWords import VocabularyFilter

def main():
	dataset = pd.read_csv('datasets/imdb.csv')
	dataset = dataset[:100]


	X = list(dataset['review'])
	
	#removes words that appear >= 60% the texts
	X = VocabularyFilter().removeWords(X, 0.6)
	print(X)
	
	count = CountVectorizer(stop_words='english')
	X = count.fit_transform(X)

	print("X.shape: ", X.shape)

	tfidf = TfidfTransformer()
	X = tfidf.fit_transform(X)

	print("X.shape: ", X.shape)

	X_train, X_test, y_train, y_test = train_test_split(X, dataset['sentiment'], test_size=0.33)

	print("X_train.shape: ", X_train.shape)
	print("X_test.shape: ", X_test.shape)

	model = MLPClassifier(max_iter=300)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	y_test = list(y_test)

	loss_values = model.loss_curve_
	acc = accuracy_score(y_test, y_pred)
	f1Score = f1_score(y_test, y_pred, pos_label='negative')
	cm = confusion_matrix(y_test, y_pred)

	print("\nAcurácia MLP", acc)
	print("F1 Score", f1Score)
	print("Matriz de Confusão")
	print(cm)

if __name__ == '__main__':
	main()