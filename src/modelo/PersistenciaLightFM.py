# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:40:44 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import os
import pickle

# Método guardar_matrices. Guardo las matrices del dataset pasado por parámetro en un archivo pickle.
def guardar_matrices(train, test, item_features, user_features, dataset):
    path = os.path.join(os.path.expanduser('~'), 'Downloads', 'matrices', 'matrices_'+dataset+'.pickle')
    matrices_pickle = open(path, 'wb')
    pickle.dump([train,test,item_features,user_features], matrices_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    matrices_pickle.close()

# Método guardar_modelo. Guardo el modelo pasado por parámetro en un archivo pickle.
def guardar_modelo(modelo, nombre_modelo, opcion_dataset):
    if opcion_dataset == 1:
        path = os.path.join(os.path.expanduser('~'), 'Downloads', 'modelos', 'modelo_movielens_'+nombre_modelo+'.pickle')
        modelo_pickle = open(path, 'wb')
        pickle.dump(modelo, modelo_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        modelo_pickle.close()
    elif opcion_dataset == 2:
        path = os.path.join(os.path.expanduser('~'), 'Downloads', 'modelos', 'modelo_anime_'+nombre_modelo+'.pickle')
        modelo_pickle = open(path, 'wb')
        pickle.dump(modelo, modelo_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        modelo_pickle.close()
    elif opcion_dataset == 3:
        path = os.path.join(os.path.expanduser('~'), 'Downloads', 'modelos', 'modelo_books_'+nombre_modelo+'.pickle')
        modelo_pickle = open(path, 'wb')
        pickle.dump(modelo, modelo_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        modelo_pickle.close()
    elif opcion_dataset == 4:
        path = os.path.join(os.path.expanduser('~'), 'Downloads', 'modelos', 'modelo_lastfm_'+nombre_modelo+'.pickle')
        modelo_pickle = open(path, 'wb')
        pickle.dump(modelo, modelo_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        modelo_pickle.close()
    else:
        path = os.path.join(os.path.expanduser('~'), 'Downloads', 'modelos', 'modelo_dating_'+nombre_modelo+'.pickle')
        modelo_pickle = open(path, 'wb')
        pickle.dump(modelo, modelo_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        modelo_pickle.close()