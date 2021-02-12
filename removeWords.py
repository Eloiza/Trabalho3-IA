class VocabularyFilter():
	def __init__(self):
		self._low_bin = 0
		self._up_bin = 0 

	#receives an frequency and returns the words with that frequency
	def _words_frequency(self, texts, frequency):
		vocabulary = {}

		for text in texts:
			word_list = text.lower().split()
			for word in word_list:
				if(not word in vocabulary.keys()):
					vocabulary[word] = 1

				else:
					vocabulary[word] +=1
	
		return vocabulary

	#Selects all words that appear in a frequency
	def _select_words(self, texts, frequency):
		words_frequency = self._words_frequency(texts, frequency)

		n_texts = len(texts)
		filtered_words = []
		for k in words_frequency.keys():
			if(words_frequency[k]/n_texts >= frequency):
				filtered_words.append(k)

		return filtered_words

	#Removes a list of words from one single text
	def _remove_words(self, text, words):
		text = text.lower().split()
		for word in words:
			i = 0 
			while (i < len(text)):
				if(text[i] == word):
					text.pop(i)
					i -= 1

				else:
					i += 1

		return text

	def _list_to_string(self, text):
		string = text.pop(0)	#initiates the string with the first word
		for word in text:
			string = string + ' ' + word

		return string

	#Removes all words from a list of texts that show up in a certain percetage
	#The percetage parameter can be a single percetage or a list of percetages
	def removeWords(self, texts, percentage):
		words_to_remove = self._select_words(texts, percentage)
		print("Total de palavras a serem removidas:", len(words_to_remove))

		for i in range(len(texts)):
			texts[i] = self._remove_words(texts[i], words_to_remove)
			texts[i] = self._list_to_string(texts[i])
		return texts

if __name__ == '__main__':
	textos = [
		'A cada de maria é branca',
		'Maria tem grande sabedoria',
		'essa Coisa é de maria',
		'cada coisa em seu lugar',
		'lugar de mulher é onde ela quiser'
	]
	
	print("Texto Original")
	print(textos)

	vocFilter = VocabularyFilter()
	new_text = vocFilter.removeWords(textos, 0.40)
	
	print("\nTexto com remoção")
	print(new_text)