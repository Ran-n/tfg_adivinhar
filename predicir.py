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
if ('-a' in __args) or ('?' in __args) or ('-h' in __args) or (len(__args) == 0):
    print('\nAxuda -----------')

    print('?/-h/-a\t\t-> Para esta mensaxe')
    print('-m\t-> Modelo a usar')
    print('-i\t-> Imaxe a adivinhar\t[adivinhar.jpg]')
    
    print("----------------\n")
    
    if len(__args) != 0:
        sys.exit()

import os
# eliminar os warnings de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import numpy as np
#-----------------------------------------------------------------


#-----------------------------------------------------------------

print('** Cargando modelo  **')
modelo = tf.keras.models.load_model('ratas/modelo')

ALTURA_IMAXE = 32
ANCHURA_IMAXE = 32
nome_clases = ['gonzales', 'speedy']

imx = keras.preprocessing.image.load_img(
    'adivinhar.jpg', target_size=(ALTURA_IMAXE, ANCHURA_IMAXE)
)

imaxe_array = keras.preprocessing.image.img_to_array(imx)
imaxe_array = tf.expand_dims(imaxe_array, 0)

prediccion = modelo.predict(imaxe_array)
puntaxe = tf.nn.softmax(prediccion[0])

clase_predita = nome_clases[np.argmax(puntaxe)]
puntaxe_predita = np.max(puntaxe)

print('{} {}% seguro'.format(clase_predita, puntaxe_predita))

#-----------------------------------------------------------------
