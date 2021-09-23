#! /usr/bin/python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------
#+ Autor:	Ran#
#+ Creado:	23/09/2021 12:20:02
#+ Editado:	23/09/2021 12:20:02
#-----------------------------------------------------------------
import sys
# meto este bloque aquí e non despois para non ter que cargar sempre tensorflow se pide axuda e así ter unha resposta rápida
# miramos se ten entradas por comando e se as ten os valores deben ser postos a tal
# quitamos a primeira
__args = sys.argv[1:]

# mensaxe de axuda
def axuda(sair):
    print('\nAxuda -----------')

    print('?|-h|-a\t -> Para esta mensaxe')
    print('-m\t -> Modelo a usar (carpeta con tódolos arquivos)')
    print('-ie\t -> Imaxe de entrada\t\t\t\t\t[adivinhar.jpg]')
    print('-is\t -> Imaxe de saída\t\t\t\t\t[mesma da de entrada]')
    print('-t\t -> Tamanho da imaxe de saída\t\t\t\t[o da orixinal]')
    print('-p\t -> Mostrar porcentaxe confianza')
    print('-c\t -> Centrar o texto na imaxe')
    
    print("----------------\n")
    
    if sair:
        sys.exit()

if ('-a' in __args) or ('?' in __args) or ('-h' in __args) or (len(__args) == 0):
    axuda(True)

import os
# eliminar os warnings de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import numpy as np
from termcolor import colored

import cv2
from PIL import ImageFont, ImageDraw, Image

from uteis import ficheiro

#-----------------------------------------------------------------

def opcions():
    if '-m' in __args:
        nome_modelo = str(__args[__args.index('-m')+1])
        modelo = tf.keras.models.load_model(nome_modelo+'/'+nome_modelo)
        print('** Cargando modelo {} **'.format(nome_modelo))
    else:
        print('!! Debes incluir o modelo a cargar !!')
        axuda(True)

    if '-ie' in __args:
        nome_imaxe = str(__args[__args.index('-ie')+1])
    else:
        nome_imaxe = 'adivinhar.jpg'
    print('** Usando imaxe "{}" **'.format(nome_imaxe))

    if '-is' in __args:
        nome_imaxe_saida = str(__args[__args.index('-is')+1])
    else:
        nome_imaxe_saida = nome_imaxe
    print('** Identificarase sobre a imaxe de entrada "{}" **'.format(nome_imaxe_saida))

    if '-p' in __args:
        porcentaxe = True
    else:
        porcentaxe = False

    if '-c' in __args:
        centro = True
    else:
        centro = False

    if '-t' in __args:
        modificar_tamanho = str(__args[__args.index('-t')+1])
    else:
        modificar_tamanho = False

    return nome_modelo, modelo, nome_imaxe, nome_imaxe_saida, porcentaxe, centro, modificar_tamanho

#-----------------------------------------------------------------

def predicir(nome_modelo, modelo, nome_imaxe):
    # cargar as dimensións das imaxes de entrenamento
    ALTURA_IMAXE_ENTRENAMENTO, ANCHURA_IMAXE_ENTRENAMENTO = ficheiro.cargarJson(nome_modelo+'/'+nome_modelo+'.parametros')['dimensións'].split('x')

    # cargar nomes das clases
    nome_clases = [clase.strip() for clase in ficheiro.cargarFich(nome_modelo+'/'+nome_modelo+'.clases')]

    imx = keras.preprocessing.image.load_img(
        nome_imaxe, target_size=(int(ALTURA_IMAXE_ENTRENAMENTO), int(ANCHURA_IMAXE_ENTRENAMENTO))
    )

    imaxe_array = keras.preprocessing.image.img_to_array(imx)
    imaxe_array = tf.expand_dims(imaxe_array, 0)

    prediccion = modelo.predict(imaxe_array)
    puntaxe = tf.nn.softmax(prediccion[0])

    clase_predita = nome_clases[np.argmax(puntaxe)]
    puntaxe_predita = np.max(puntaxe)

    print('\n********************************************************')
    print(' A imaxe é da clase {} cunha confianza do {}'.format(colored(clase_predita, 'white', 'on_red'), 
        colored(str(puntaxe_predita*100)[:5]+'%', 'white', 'on_blue')))
    print('********************************************************')

    return clase_predita, puntaxe_predita


def editar_imaxe(nome_imaxe, nome_imaxe_saida, clase_predita, puntaxe_predita, porcentaxe, centro, modificar_tamanho):
    puntaxe_predita = str(puntaxe_predita*100)[:5]+'%'

    imx = cv2.imread(nome_imaxe)

    if modificar_tamanho:
        x, y = (modificar_tamanho.split('x'))
        imx = cv2.resize(imx, (int(x), int(y)))

    fonte = cv2.FONT_HERSHEY_SIMPLEX
    
    tamanho_texto1 = cv2.getTextSize(clase_predita, fonte, 1, 2)[0]
    x_texto1 = int((imx.shape[1] - tamanho_texto1[0]) / 2)
    if centro and porcentaxe:
        y_texto1 = int((imx.shape[0] - tamanho_texto1[1]) / 2)
    
    elif centro:
        y_texto1 = int((imx.shape[0] + tamanho_texto1[1]) / 2)
    
    else:
        y_texto1 = int((0 + tamanho_texto1[1]))

    cv2.putText(imx, clase_predita, (x_texto1, y_texto1), fonte, 1, (255, 255, 255), 2)

    if porcentaxe:
        tamanho_texto2 = cv2.getTextSize(puntaxe_predita, fonte, 1, 2)[0]
        x_texto2 = int((imx.shape[1] - tamanho_texto2[0]) / 2)
        
        if centro:
            y_texto2 = int(y_texto1 + 2*tamanho_texto1[1])
        else:
            y_texto2 = int((imx.shape[0] - tamanho_texto2[1] / 2))
        cv2.putText(imx, puntaxe_predita, (x_texto2, y_texto2), fonte, 1, (255, 255, 255), 2)

    cv2.imwrite(nome_imaxe_saida, imx)

    cv2.destroyAllWindows()

#-----------------------------------------------------------------

def main():
    # collemos as opcións introducidas por comandos
    nome_modelo, modelo, nome_imaxe, nome_imaxe_saida, porcentaxe, centro, modificar_tamanho = opcions()
    clase_predita, puntaxe_predita = predicir(nome_modelo, modelo, nome_imaxe)
    editar_imaxe(nome_imaxe, nome_imaxe_saida, clase_predita, puntaxe_predita, porcentaxe, centro, modificar_tamanho)

#-----------------------------------------------------------------

if __name__=="__main__":
    main()

#-----------------------------------------------------------------
