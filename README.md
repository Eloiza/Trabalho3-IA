# Trabalho3-IA
Implementação do terceiro trabalho da disciplina de Inteligência artificial

### Descrição
O objetivo deste trabalho era aprender mais sobre o processamento de linguagem natural (NLP). Para isto foi estudado os algoritmos Bag of Words (BOW), Term Frequency-Inverse Document Frequency (TF-IDF) e Word to Vector (Word2Vec). Parte deste trabalho foi baseado em uma implementação destes algoritmos disponíveis [aqui!](https://github.com/dimmykarson/aulanlp). 
Neste trabalho foi pedido para que se implementasse um módulo para remover palavras da base de dados usada dependendo de uma frequência passada. Para isso foi desenvolvido o módulo `removeWords.py` que implementa uma classe com essa feature descrita. Após remover tais palavras da base era necessário testar o desempenho de uma rede neural Multi-Layer Perceptron (MLP), a pasta `Results` apresenta o log dos resultados obtidos com os testes.

### Dependências 
Para executar o código deste repositório é necessário possui os módulos: pandas, scikit-learn, gensim. Abaixo há um guia rápido de instalação :smile:

##### Instalando biblioteca Pandas
```
pip3 -U install pandas 
```

##### Instalando biblioteca Scikit-learn
```
pip3 -U install scikit-learn 
```

##### Instalando biblioteca Gensim
```
pip3 -U install  gensim 
```

### Executando o código
##### Para executar o algoritmo BOW + TFIDF
```
python3 bow_tfidf.py
```
##### Para executar o algoritmo Word2Vect
```
python3 w2v.py
```
##### Para executar demo do módulo removeWords
```
python3 removeWords.py
```
