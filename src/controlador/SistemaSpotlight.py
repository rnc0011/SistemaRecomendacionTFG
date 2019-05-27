# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:40:22 2019

@author: Raúl
"""

# Importo todo lo necesario
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\Raúl\\Google Drive\\GitHub\\SistemaRecomendacionTFG\\src\\modelo')
import Entrada
import Salida

from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score, mrr_score, precision_recall_score
#from spotlight.evaluation import mrr_score
#from spotlight.evaluation import precision_recall_score

# Clase SistemaSpotlight.
class SistemaSpotlight:
    
    # Variables globales
    global train_e
    global test_e
    global train_i
    global test_i
    global modelo
    
    # Método __init__. Inicializa la clase con el conjunto de datos escogido.
    def __init__(self, opcion_dataset, opcion_modelo):
        self.opcion_dataset = opcion_dataset
        self.opcion_modelo = opcion_modelo
     
    # Método interacciones_movielens. Crea las interacciones de movielens con las que poder utilizar los modelos.        
    def interacciones_movielens(self):
        global train_e
        global test_e  
        global train_i
        global test_i
    
        # Leo los csv
        Entrada.leer_movielens()
        
        # Obtengo los arrays con los ids de los usuarios y los items
        users_ids = np.asarray(Entrada.ratings_df['Id Usuario'].tolist(), dtype=np.int32)
        with np.nditer(users_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
                
        items_ids = np.asarray(Entrada.ratings_df['Id Película'].tolist(), dtype=np.int32)
        with np.nditer(items_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
          
        # Obtengo el array con las valoraciones    
        ratings = np.asarray(Entrada.ratings_df['Valoración'].tolist(), dtype=np.float32)
        
        # Obtengo las interacciones
        interacciones_e = Interactions(users_ids, items_ids, ratings, num_users=len(np.unique(users_ids)), num_items=len(np.unique(items_ids)))
        interacciones_i = Interactions(users_ids, items_ids, num_users=len(np.unique(users_ids)), num_items=len(np.unique(items_ids)))
        
        # Divido las interacciones en train y test
        train_e, test_e = random_train_test_split(interacciones_e)
        train_i, test_i = random_train_test_split(interacciones_i)
        
    # Método interacciones_anime. Crea las interacciones de movielens con las que poder utilizar los modelos.        
    def interacciones_anime(self):
        global train_e
        global test_e       
    
        # Leo los csv
        Entrada.leer_anime()
        
        # Obtengo los arrays con los ids de los usuarios y los items
        users_ids = np.asarray(Entrada.ratings_df['Id Usuario'].tolist(), dtype=np.int32)
        with np.nditer(users_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
                
        items_ids = np.asarray(Entrada.ratings_df['Id Anime'].tolist(), dtype=np.int32)
        with np.nditer(items_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
          
        # Obtengo el array con las valoraciones    
        ratings = np.asarray(Entrada.ratings_df['Valoración'].tolist(), dtype=np.float32)
        
        # Obtengo las interacciones
        interacciones = Interactions(users_ids, items_ids, ratings, num_users=len(np.unique(users_ids)), num_items=len(np.unique(items_ids)))
        
        # Divido las interacciones en train y test
        train, test = random_train_test_split(interacciones)
        
    # Método interacciones_book_crossing. Crea las interacciones de movielens con las que poder utilizar los modelos.        
    def interacciones_book_crossing(self):
        global train
        global test       
    
        # Leo los csv
        Entrada.leer_book_crossing()
        
        # Obtengo los arrays con los ids de los usuarios y los items
        users_ids = np.asarray(Entrada.ratings_df['Id Usuario'].tolist(), dtype=np.int32)
        with np.nditer(users_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
                
        items_ids = np.asarray(Entrada.ratings_df['ISBN'].tolist(), dtype=np.int32)
        with np.nditer(items_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
          
        # Obtengo el array con las valoraciones    
        ratings = np.asarray(Entrada.ratings_df['Valoración'].tolist(), dtype=np.float32)
        
        # Obtengo las interacciones
        interacciones = Interactions(users_ids, items_ids, ratings, num_users=len(np.unique(users_ids)), num_items=len(np.unique(items_ids)))
        
        # Divido las interacciones en train y test
        train, test = random_train_test_split(interacciones)
        
    # Método interacciones_lastfm. Crea las interacciones de movielens con las que poder utilizar los modelos.        
    def interacciones_lastfm(self):
        global train
        global test       
    
        # Leo los csv
        Entrada.leer_lastfm()
        
        # Obtengo los arrays con los ids de los usuarios y los items
        users_ids = np.asarray(Entrada.ratings_df['Id Usuario'].tolist(), dtype=np.int32)
        with np.nditer(users_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
                
        items_ids = np.asarray(Entrada.ratings_df['Id Artista'].tolist(), dtype=np.int32)
        with np.nditer(items_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
          
        # Obtengo el array con las valoraciones    
        ratings = np.asarray(Entrada.ratings_df['Veces escuchado'].tolist(), dtype=np.float32)
        
        # Obtengo las interacciones
        interacciones = Interactions(users_ids, items_ids, ratings, num_users=len(np.unique(users_ids)), num_items=len(np.unique(items_ids)))
        
        # Divido las interacciones en train y test
        train, test = random_train_test_split(interacciones)
        
    # Método interacciones_dating_agency. Crea las interacciones de movielens con las que poder utilizar los modelos.        
    def interacciones_dating_agency(self):
        global train
        global test       
    
        # Leo los csv
        Entrada.leer_dating_agency()
        
        # Obtengo los arrays con los ids de los usuarios y los items
        users_ids = np.asarray(Entrada.ratings_df['Id Usuario'].tolist(), dtype=np.int32)
        with np.nditer(users_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
                
        items_ids = np.asarray(Entrada.ratings_df['Id Match'].tolist(), dtype=np.int32)
        with np.nditer(items_ids, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x - 1
          
        # Obtengo el array con las valoraciones    
        ratings = np.asarray(Entrada.ratings_df['Valoración'].tolist(), dtype=np.float32)
        
        # Obtengo las interacciones
        interacciones = Interactions(users_ids, items_ids, ratings, num_users=len(np.unique(users_ids)), num_items=len(np.unique(items_ids)))
        
        # Divido las interacciones en train y test
        train, test = random_train_test_split(interacciones)
        
    # Método obtener_interacciones. Crea las interacciones con las que poder utilizar los modelos en función del dataset escogido.    
    def obtener_interacciones(self):
        if self.opcion_dataset == 'movielens':
            self.interacciones_movielens()
        elif self.opcion_dataset == 'anime':
            self.interacciones_anime()
        elif self.opcion_dataset == 'books':
            self.interacciones_book_crossing()
        elif self.opcion_dataset == 'lastfm':
            self.interacciones_lastfm()
        else:
            self.interacciones_dating_agency()
            
    # Método modelo_factorizacion_explicito. Crea el modelo de factorización explícito.
    def modelo_factorizacion_explicito(self):
        global train_e
        global modelo
        
        # Obtengo y entreno el modelo
        modelo = ExplicitFactorizationModel(loss='logistic', use_cuda=True)
        modelo.fit(train_e, verbose=True)
        
    # Método modelo_factorizacion_implicito. Crea el modelo de factorización implícito.
    def modelo_factorizacion_implicito(self):
        global train_i
        global modelo
        
        # Obtengo y entreno el modelo
        modelo = ImplicitFactorizationModel(loss='bpr', use_cuda=True)
        modelo.fit(train_i, verbose=True)
            
    # Método obtener_modelos. Crea los modelos en función de la opción escogida.
    def obtener_modelos(self):
        if self.opcion_modelo == 1:
            self.modelo_factorizacion_explicito()
        elif self.opcion_modelo == 2:
            self.modelo_factorizacion_implicito()
        
    # Método resultados_factorizacion_explicito. Calcula las métricas del modelo de factorización explícito.
    def resultados_factorizacion_explicito(self):
        global train_e
        global test_e
        global modelo
        
        # Calculo las métricas
        rmse = rmse_score(modelo, test_e)
        mrr = mrr_score(modelo, test_e, train=train_e).mean()
        precision, recall = precision_recall_score(modelo, test_e, train=train_e, k=10)
        
        Salida.imprimir_resultados_dl(mrr, precision.mean(), recall.mean(), rmse)
        
    # Método resultados_factorizacion_implicito. Calcula las métricas del modelo de factorización implícito.
    def resultados_factorizacion_implicito(self):
        global train_i
        global test_i
        global modelo
        
        # Calculo las métricas
        #rmse = rmse_score(modelo, test_i)
        mrr = mrr_score(modelo, test_i, train=train_i).mean()
        precision, recall = precision_recall_score(modelo, test_i, train=train_i, k=10)
        
        Salida.imprimir_resultados_dl(mrr, precision.mean(), recall.mean())
        
    # Método obtener_resultados. Calcula las métricas en función del modelo escogido.
    def obtener_resultados(self):
        if self.opcion_modelo == 1:
            self.resultados_factorizacion_explicito()
        elif self.opcion_modelo == 2:
            self.resultados_factorizacion_implicito()
    
    
    