# -*- coding: utf-8 -*-
#   --------DANIEL CRUZ PAZ----------
import nltk
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import math

#Se calcula el coseno entre el documento actual y los documentos de la coleccion.
#La funcion recibe la matriz(Binaria,TF,TfxIDF) y una lista de los titulos de la coleccion.
def similitud( matriz , textos , vectorQ, query):
    dim = matriz.shape
    sum = 0
    listaSim = []
    print "Documentos con similitud al documento: ", query
    #Se recorren todos los documentos de la coleccion.(Filas de la Matriz)
    for x in range(0,dim[0]):
        sum = 0
        q = 0.0
        d = 0.0
        #Se recorren todas las palabras del vocabulario(Columnas de la Matriz)
        for y in range(0,dim[1]):
            #Sumatorias
            sum = sum + vectorQ[y] * matriz[x][y]
            q = q + pow(vectorQ[y],2)
            d = d + pow(matriz[x][y],2)
        cos = sum / float(math.sqrt(q * d))
        #Se agrega una lista con los indices del documento y su coseno.
        listaSim.append([ x , cos ])
    #Se ordena la lista anterior en orden descendente.
    listaSim.sort(key = lambda columna: columna[1], reverse = True)
    n = 0
    listaRecu = []
    for k in range(0,len(listaSim)):
        if( listaSim[k][1] > 0):
            #print textos[listaSim[k][0]]
            listaRecu.append(listaSim[k][0]+1)
            n = n + 1
    print "Numero de docs regresados: " , n
    return listaRecu

#/////////////////////////////////////////////////////////////////////////////////////////////////////////

def grafica( listaRecB,listaRecTF, listaRecTFIDF, listaRelevantes):
    binarioP = []
    binarioR = []
    TFP = []
    TFR = []
    TFIDFP = []
    TFIDFR = []
    aux = 0
    r = 0
    #print listaRecB
    #print listaRelevantes
    for i in range(0,len(listaRecB)):
        if(listaRelevantes[1].count(str(listaRecB[i]))):
            r = r + 1
            binarioP.append( r / float(i+1) )
            aux = aux + ( 1 / float(len(listaRelevantes[1])) )
            binarioR.append( aux )
    aux = 0
    r = 0
    for i in range(0,len(listaRecTF)):
        if(listaRelevantes[1].count(str(listaRecTF[i]))):
            r = r + 1
            TFP.append( r / float(i+1) )
            aux = aux + ( 1 / float(len(listaRelevantes[1])) )
            TFR.append( aux )
    aux = 0
    r = 0
    for i in range(0,len(listaRecTFIDF)):
        if(listaRelevantes[1].count(str(listaRecTFIDF[i]))):
            r = r + 1
            TFIDFP.append( r / float(i+1) )
            aux = aux + ( 1 / float(len(listaRelevantes[1])) )
            TFIDFR.append( aux )

    print "Relevantes" , r   
    
    txt = "Graph: " +  listaRelevantes[0] + "\n"
    plt.title(txt)
    plt.xlabel("Recall")   # Inserta el título del eje X 
    plt.ylabel("Precision")   # Inserta el título del eje Y
    plt.plot(binarioR, binarioP, marker='x', linestyle=':', label = "Binario")
    plt.plot(TFR, TFP, 'o-', label = "TF")
    plt.plot(TFIDFR, TFIDFP, marker='D', linestyle='--', label = "TF-IDF")

    plt.grid(True)
    plt.legend()
    plt.show()        


#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Extraer querys de la coleccion Querys.text
def extraerRelevantesQuery( doc  ):
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
    #similitud(binaria, textos)
    return binaria

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
    #similitud( matriztf , textos)
    return matriztf

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
    #similitud( matrizidf , textos )
    return matrizidf
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Vector Query con pesos
def pesoQuery( textos, vocabulario, listaFrecu, query):
    y = 0
    numDocs = len(textos)
    matrizQ = np.zeros((3,len(vocabulario)))
    for word in vocabulario:
        if query.count(word) > 0:
            matrizQ[0,y] = 1
        matrizQ[1,y] = query.count(word)
        matrizQ[2,y] = query.count(word) * np.log((numDocs/listaFrecu[y][2]))
        y = y + 1
        
    return matrizQ
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
def extraerQuery( doc , stem  ):
    ps = PorterStemmer()
    var = ""
    #Lista para guardar las querys
    querys = []
    query = []
    #Bandera
    band = 0
    for word in doc:
        if word == ".W" or word ==".w":
            band = 1
        if (word == ".N" or word == ".n" or word == '.A' or word=='.a') and band == 1:
            #Agregar titulo a la lista de titulos
            querys.append(query)
            query = []
            var = ""
            band = 0
        if band == 1:
            if word == ".W":
                word = word.lower()
            if word != ".w" and word != ".":
                if stem == True:
                    word = word.lower()
                    word = ps.stem(word)
                query.append(word)
    
    print "Numero de Documentos: "
    print len(querys)

    return querys
        
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Encuentra las frecuencias de cada palabra en la coleccion y el # de documentos en los que aparece.
def funcionFrecuencias( vocabulario, textos):
    #Lista de Frecuencias de palabras en Coleccion y Documentos.
    #listaFrec[ palabra , Frecuencia de palabra en Coleccion ,  Frecuencia de palabra en Documentos]
    listaFrec = []
    for palabra in vocabulario:
        fPalabra = 0
        fDocs = 0
        for titulo in textos:
            num = titulo.count(palabra)
            fPalabra = fPalabra + num
            if num > 0:
                fDocs = fDocs + 1
        listaFrec.append([palabra,fPalabra,fDocs])
    
    # print "Vocabulario\t|Frecuencia en Coleccion\t|Frecuencia en Documentos"
    #Ordena la lista en orden decreciente de acuerdo a la Frecuencia de la palabra en la Coleccion
    #listaFrec.sort(key=lambda palabra: palabra[1],reverse=True)
    
    return listaFrec

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Funcion principal
def funcionTarea( doc , stem, docQuerys ):
    ps = PorterStemmer()
    var = ""
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
            var = ""
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
    
    #Obtener el vocabulario de la lista de palabras.
    vocabulario = set(aux)
    #Frecuencias
    listaFrecu = funcionFrecuencias(vocabulario , textos)
    #Matrizes
    MBinaria = obtenerMatrizBinaria( textos, vocabulario)
    MTf = obtenerMatrizTF(textos,vocabulario)
    MIdf = obtenerMatrizTFIDF(textos,vocabulario,listaFrecu)
    
    #Extraer Querys
    docQrels = open("/Users/DanniC/Documents/INAOE_2Cuatri/RecuperacionInformacion/Colecciones/cacm/qrels.text","r")
    qrels = extraerRelevantesQuery( docQrels)
    #Realizar Stemming a las querys "True" ( Solo si a la coleccion se le esta haciendo stemming )
    querys = extraerQuery( docQuerys, True)
    
    masR = [51, 50, 49, 48, 47]
    menosR = [0, 1, 2, 3, 4, 5]

    #Querys Mas Relevantes
    for i in range(0,5):
        q = int(qrels[masR[i]][0]) - 1
        print "Query : ", q + 1
        matrizQue = pesoQuery( textos, vocabulario, listaFrecu , querys[q] )
        recuperadosB = similitud( MBinaria , textos , matrizQue[0,] , querys[q] )
        recuperadosTF = similitud( MTf , textos , matrizQue[1,] , querys[q] )
        recuperadosTFIDF = similitud( MIdf , textos , matrizQue[2,] , querys[q] )
        grafica( recuperadosB, recuperadosTF, recuperadosTFIDF, qrels[masR[i]] )
        print qrels[masR[i]]

    
    #Querys Mas Relevantes
    for i in range(0,5):
        q = int(qrels[menosR[i]][0]) - 1
        print "Query : ", q + 1
        matrizQue = pesoQuery( textos, vocabulario, listaFrecu , querys[q] )
        recuperadosB = similitud( MBinaria , textos , matrizQue[0,] , querys[q] )
        recuperadosTF = similitud( MTf , textos , matrizQue[1,] , querys[q] )
        recuperadosTFIDF = similitud( MIdf , textos , matrizQue[2,] , querys[q] )
        grafica( recuperadosB, recuperadosTF, recuperadosTFIDF, qrels[menosR[i]] )
        print qrels[menosR[i]]



    return

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#Leer texto CACM
#docOriginal = nltk.Text(nltk.regexp_tokenize(open("/Users/DanniC/Documents/INAOE_2Cuatri/RecuperacionInformacion/Colecciones/cacm/cacm.all","r").read(),"[A-Za-z'\.]+"))
docPreprocesado = nltk.Text(nltk.regexp_tokenize(open("/Users/DanniC/Documents/INAOE_2Cuatri/RecuperacionInformacion/Colecciones/cacm/cacm.all","r").read().lower(),"[a-z'\.]+"))
docQuerys = nltk.Text(nltk.regexp_tokenize(open("/Users/DanniC/Documents/INAOE_2Cuatri/RecuperacionInformacion/Colecciones/cacm/query.text","r").read(),"[A-Za-z'\.]+"))

#/////////////////////////////////////////////////////////////////////////////////////////////////////////

#QUERYS
#Documento sin realizar Stemming ( Comentado)
#funcionTarea(docOriginal, False, docQuerys)

#Documento realizando Stemming
funcionTarea(docPreprocesado, True, docQuerys)


