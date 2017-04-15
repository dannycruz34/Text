import nltk
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import math

def extraerQuery( doc  ):
    dic = {}
    for line in doc.readlines():
        pline = line.split(" ")
        if( pline[0] in dic.keys()):
            dic[pline[0]] += [pline[1]]
        else:
            dic[pline[0]] = []
            dic[pline[0]] += [pline[1]]
    ordenada= dic.items()
    ordenada.sort(key=lambda x: len(x[1]))
    return ordenada
    

docQrels = open("/Users/DanniC/Documents/INAOE_2Cuatri/RecuperacionInformacion/Colecciones/cacm/qrels.text","r")

qrels = extraerQuery( docQrels)
print qrels[0][0]
j = 0
for i in qrels:
    j = j + 1

print j
import matplotlib.pyplot as plt
import numpy as np
lista1 = [11,2,3,15,8,13,21,34]
lista2 = [2,3,4,2,3,6,4,10]
lista3 = [9,15,9,15,9,15,9,15]
plt.plot(lista1)
plt.plot(lista2) 
plt.plot(lista3)
plt.title("Graph")
plt.xlabel("Recall")   # Inserta el título del eje X 
plt.ylabel("Precision")   # Inserta el título del eje Y
plt.plot(lista1, label = "Binario")
plt.plot(lista2, label = "TF")
plt.plot(lista3, label = "TfxIDF")
plt.grid(True)
plt.legend()
plt.show()


binaria = np.zeros((5))
print binaria
