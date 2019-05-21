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
    path = os.path.join(os.path.expanduser('~'), 'Downloads', 'persistencia', 'matrices', 'matrices_'+dataset+'.pickle')
    matrices_pickle = open(path, 'wb')
    pickle.dump([train,test,item_features,user_features], matrices_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    matrices_pickle.close()

# Método guardar_modelo. Guardo el modelo pasado por parámetro en un archivo pickle.
def guardar_modelo_clasico(modelo, nombre_modelo, opcion_dataset):
    path = os.path.join(os.path.expanduser('~'), 'Downloads', 'persistencia', 'modelos_clasicos', 'modelo_'+opcion_dataset+'_'+nombre_modelo+'.pickle')
    modelo_pickle = open(path, 'wb')
    pickle.dump(modelo, modelo_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    modelo_pickle.close()


