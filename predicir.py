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

    print('?/-h/-a\t\t-> Para esta mensaxe')
    print('-m\t\t-> Modelo a usar (carpeta con tódolos arquivos)')
    print('-i\t\t-> Imaxe a adivinhar\t[adivinhar.jpg]')
    
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

    if '-i' in __args:
        nome_imaxe = str(__args[__args.index('-i')+1])
    else:
        nome_imaxe = 'adivinhar.jpg'
    print('** Usando imaxe "{}" **'.format(nome_imaxe))

    return nome_modelo, modelo, nome_imaxe

#-----------------------------------------------------------------

def predicir():
    # collemos as opcións introducidas por comandos
    nome_modelo, modelo, nome_imaxe = opcions()

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
    print(' A imaxe é da clase {} cunha confianza do {}'.format(colored(clase_predita, 'white', 'on_red'), colored(str(puntaxe_predita*100)[:5]+'%', 'white', 'on_blue')))
    print('********************************************************')

def editar_imaxe():
    print('a')


#-----------------------------------------------------------------

def main():
    predicir()
    editar_imaxe()

#-----------------------------------------------------------------

if __name__=="__main__":
    main()

#-----------------------------------------------------------------
