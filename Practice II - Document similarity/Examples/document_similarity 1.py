import pandas as pd
import math 


def cosine(x, y):
	val = sum(x[index] * y[index] for index in range(len(x)))
	sr_x = math.sqrt(sum(x_val**2 for x_val in x))
	sr_y = math.sqrt(sum(y_val**2 for y_val in y))
	res = val/(sr_x*sr_y)
	return (res)


                     #~ computación  biología  artes   informática
#~ computadora          300       100     25     270
#~ internet             200        50     70     300
#~ célula                10       250      3     8
#~ pintura                2        10    280     6
#~ data_words = {'computadora': [300, 100, 25], 'internet': [200, 50, 70], 'célula': [10, 250, 3], 'pintura': [2, 10, 280]}
data_documents = {'computación': [300, 200, 10, 2], 'biología': [100, 50, 250, 10], 'artes': [25, 70, 3, 280], 'informática': [270, 300, 8, 6]}

#~ print ('Word Similarity')
#~ cosine(data_words['computadora'], data_words['internet'])
#~ cosine(data_words['computadora'], data_words['célula'])
#~ cosine(data_words['computadora'], data_words['pintura'])

print ('Document Similarity')
print ('computación - biología {}'.format( cosine(data_documents['computación'], data_documents['biología'])))
print ('computación - artes {}'.format(cosine(data_documents['computación'], data_documents['artes'])))
print('biología - artes {}'.format(cosine(data_documents['biología'], data_documents['artes'])))
print('computación - informática {}'.format(cosine(data_documents['computación'], data_documents['informática'])))







