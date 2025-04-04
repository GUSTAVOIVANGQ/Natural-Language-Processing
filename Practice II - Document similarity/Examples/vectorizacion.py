from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os.path
import sys
import pickle


corpus = ['El niño corre velozmente por el camino a gran velocidad .',
          'El coche rojo del niño es grande .',
          'El coche tiene un color rojo brillante y tiene llantas nuevas .',
          '¿ Las nuevas canicas del niño son color rojo ?'
]


#Representación vectorial por frecuencia
# ~ vectorizador_frecuencia = CountVectorizer()
vectorizador_frecuencia = CountVectorizer(token_pattern= r'(?u)\w+|\w+\n|\.|\¿|\?')
X = vectorizador_frecuencia.fit_transform(corpus)
print (vectorizador_frecuencia.get_feature_names_out())
print (X)#sparse matrix
# ~ print (type(X))#sparse matrix
# ~ print (type(X.toarray()))#dense ndarray

print('Representación vectorial por frecuencia')
print (X.toarray())


# Representación vectorial binarizada
vectorizador_binario = CountVectorizer(binary=True)
vectorizador_binario = CountVectorizer(binary=True, token_pattern= r'(?u)\w+|\w+\n|\.|\¿|\?')
X = vectorizador_binario.fit_transform(corpus)
print ('Representación vectorial binarizada')
print (X.toarray())#dense ndarray


# ~ #Representación vectorial tf-idf
vectorizador_tfidf = TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.|\¿|\?')
X = vectorizador_tfidf.fit_transform(corpus)
print ('Representación vectorial tf-idf')
print (X.toarray())


vectorizador_tfidf = TfidfVectorizer(token_pattern= r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range = (2,2))
X = vectorizador_tfidf.fit_transform(corpus)
print (vectorizador_tfidf.get_feature_names_out())
print ('Representación vectorial tf-idf')
print (X.toarray())


vectorizador_tfidf = TfidfVectorizer(token_pattern= r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range = (1,3))
X = vectorizador_tfidf.fit_transform(corpus)
print (vectorizador_tfidf.get_feature_names_out())
print ('Representación vectorial tf-idf')
print (X.toarray())

#Use of Pickle
if (os.path.exists('X_vectorizador_tfidf.pkl')):
	print ('Ya existe')
	vector_file = open ('X_vectorizador_tfidf.pkl','rb')
	X = pickle.load(vector_file)
else:
	vector_file = open ('X_vectorizador_tfidf.pkl','wb')
	vectorizador_tfidf = TfidfVectorizer(token_pattern= r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range = (1,3))
	X = vectorizador_tfidf.fit_transform(corpus)
	pickle.dump(X, vector_file)
	vector_file.close()

print ('Representación vectorial tf-idf')
print (X.toarray())
