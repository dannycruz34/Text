# -*- coding: utf-8 -*-
#   --------DANIEL CRUZ PAZ----------
import nltk
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import bigrams
import numpy as np
import matplotlib.pyplot as plt
import math

#Se calcula el coseno entre el documento actual y los documentos de la coleccion.
#La funcion recibe la matriz(Binaria,TF,TfxIDF) y una lista de los titulos de la coleccion.
def similitud( matriz , textos):
    dim = matriz.shape
    sum = 0
    for i in range(0,3):
        listaSim = []
        docN = "Documentos con similitud al documento :"
        #Se recorren todos los documentos de la coleccion.(Filas de la Matriz)
        for x in range(0,dim[0]):
            sum = 0
            q = 0.0
            d = 0.0
            #Se recorren todas las palabras del vocabulario(Columnas de la Matriz)
            for y in range(0,dim[1]):
                #Sumatorias
                sum = sum + matriz[i][y] * matriz[x][y]
                q = q + pow(matriz[i][y], 2)
                d = d + pow(matriz[x][y], 2)
            cos = sum / float(math.sqrt(q * d))
            #Se agrega una lista con los indices del documento y su coseno.
            listaSim.append([ x , cos ])
        #Se ordena la lista anterior en orden descendente.
        listaSim.sort(key = lambda columna: columna[1], reverse = True)
        t = textos[listaSim[0][0]]
        for i in range(0, len(t)):
                docN = docN + " " + t[i][0]
        print docN + "\n"
        
        for k in range(1, 6):
            s = ""
            s = s + str(listaSim[k][0]) + ":" + str(listaSim[k][1])
            t = textos[listaSim[k][0]]
            for i in range(0,len(t)):
                s = s + " " + t[i][0]
            print s
        print "\n"

    return

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Se obtiene la Matriz con peso Binario
def obtenerMatrizBinaria( textos , vocabulario ):
    binaria = np.zeros((len(textos), len(vocabulario)))
    x = 0
    for titulo in textos:
        y = 0
        for word in vocabulario:
            if titulo.count(word) > 0:
                binaria[x,y] = 1
            y = y + 1
        x = x + 1
    #Se llama a la funcion similitud para calcular el coseno    
    similitud(binaria, textos)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Se obtiene la Matriz con peso Tf
def obtenerMatrizTF( textos , vocabulario ):
    matriztf = np.zeros((len(textos), len(vocabulario)))
    x = 0
    for titulo in textos:
        tamT = len(titulo)
        y = 0
        for word in vocabulario:
            matriztf[x,y] = titulo.count(word)
            y = y + 1
        x = x + 1
    #Se llama a la funcion similitud para calcular el coseno    
    similitud( matriztf , textos)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Se obtiene la Matriz con peso TfxIdf
def obtenerMatrizTFIDF( textos , vocabulario , listaFrecu):
    matrizidf = np.zeros((len(textos), len(vocabulario)))
    x = 0
    numDocs = len(textos)
    for titulo in textos:
        y = 0
        for word in vocabulario:
            matrizidf[x,y] = titulo.count(word) * np.log((numDocs/listaFrecu[y][2]))
            y = y + 1
        x = x + 1
    #Se llama a la funcion similitud para calcular el coseno.    
    similitud( matrizidf , textos )
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Encuentra las frecuencias de cada palabra en la coleccion y el # de documentos en los que aparece.
def funcionFrecuencias(  textos, vocabulario ):
    #Lista de Frecuencias de palabras en Coleccion y Documentos.
    #listaFrec[ palabra , Frecuencia de palabra en Coleccion ,  Frecuencia de palabra en Documentos]
    listaFrec = []
    for palabra in vocabulario:
        fPalabra = 0
        fDocs = 0
        for titulo in textos:
            num = titulo.count( palabra )
            fPalabra = fPalabra + num
            if num > 0:
                fDocs = fDocs + 1
        listaFrec.append([palabra,fPalabra,fDocs])
    
    # print "Vocabulario\t|Frecuencia en Coleccion\t|Frecuencia en Documentos"
    #Ordena la lista en orden decreciente de acuerdo a la Frecuencia de la palabra en la Coleccion
    #listaFrec.sort(key=lambda palabra: palabra[1],reverse=True)
    
    return listaFrec
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
# Realizar el etiquetado de titulos
def hacerTag( textos):
    auxNuevo = []
    textosTag = []
    salida = []
    for titulo in textos:
        text = nltk.pos_tag(titulo)
        textosTag.append(text)
        auxNuevo.extend(text)
        
    vocabulario = set(auxNuevo)
    salida.append([textosTag,vocabulario])
    print len(vocabulario)
    return salida
    
    
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
# Realizar el indexado por bigramas
def hacerBigramas( textos ):
    auxNuevo = []
    textosBi = []
    salida = []
    for titulo in textos:
        text = []
        text = titulo
        text.extend( list(bigrams(titulo)) )
        textosBi.append(text)
        auxNuevo.extend(text)

    vocabulario = set(auxNuevo)
    print len(vocabulario)
    salida.append([textosBi,vocabulario])
    return salida
        
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Funcion principal
def funcionTarea( doc , stem):
    ps = PorterStemmer()
    #Lista para guardar los titulos
    textos = []
    #Lista para guardar las palabras de la coleccion(titulos)
    aux = []
    titulo = []
    #Bandera
    band = 0
    for word in doc:
        if word == ".T" or word ==".t":
            band = 1
        if (word == ".B" or word == ".W" or word == '.A' or word=='.w' or word == '.b' or word == '.a') and band == 1:
            #Agregar titulo a la lista de titulos
            textos.append(titulo)
            titulo = []
            band = 0
        if band == 1:
            if word == ".T":
                word = word.lower()
            if word != ".t" and word != ".":
                #Si la funcion realizara stemming a las palabras
                if stem == True:
                    word = ps.stem(word)
                titulo.append(word)
                #Agregar palabras a la lista de palabras.
                aux.append(word)
    
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#**************************************** POS TAG ************************************
    #POS TAG
    print "-------- POS TAG -----------"
    salida = hacerTag(textos)
    #Frecuencias etiquetadas
    listaFrecu = funcionFrecuencias(salida[0][0], salida[0][1])
    #Matrizes
    #salida[0[0]   TITULOS ETIQUETADOS
    #salida[0][1]  VOCABULARIO ETIQUETADO
    print "MATRIZ BINARIA\n"
    obtenerMatrizBinaria( salida[0][0], salida[0][1] )
    print "MATRIZ TF\n"
    obtenerMatrizTF( salida[0][0], salida[0][1] )
    print "MATRIZ TF-IDF\n"
    obtenerMatrizTFIDF( salida[0][0], salida[0][1] , listaFrecu )
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    #INDEXADO POR BIGRAMAS
    print "-------- INDEXADO POR BIGRAMAS -----------\n"
    salida = hacerBigramas(textos)
    #Frecuencias etiquetadas
    listaFrecu = funcionFrecuencias(salida[0][0], salida[0][1])
    #Matrizes
    #salida[0[0]   TITULOS ETIQUETADOS
    #salida[0][1]  VOCABULARIO ETIQUETADO
    print "MATRIZ BINARIA\n"
    obtenerMatrizBinaria( salida[0][0], salida[0][1] )
    print "MATRIZ TF\n"
    obtenerMatrizTF( salida[0][0], salida[0][1] )
    print "MATRIZ TF-IDF\n"
    obtenerMatrizTFIDF( salida[0][0], salida[0][1] , listaFrecu )

    return

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Leer texto CACM
docOriginal = nltk.Text(nltk.regexp_tokenize(open("/Users/DanniC/Documents/INAOE_2Cuatri/RecuperacionInformacion/Colecciones/cacm/cacm.all","r").read(),"[A-Za-z'\.]+"))
#docPreprocesado = nltk.Text(nltk.regexp_tokenize(open("/Users/DanniC/Documents/INAOE_2Cuatri/RecuperacionInformacion/Colecciones/cacm/cacm.all","r").read().lower(),"[a-z'\.]+"))

#/////////////////////////////////////////////////////////////////////////////////////////////////////////

#QUERYS
#Documento sin realizar Stemming ( Comentado)
funcionTarea(docOriginal, False )

#Documento realizando Stemming
#funcionTarea(docPreprocesado, True)


