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
from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
from spotlight.evaluation import mrr_score
from spotlight.evaluation import precision_recall_score

# Clase SistemaSpotlight.
class SistemaSpotlight:
    
    # Variables globales
    global train
    global test
    global modelo
    
    # Método __init__. Inicializa la clase con el conjunto de datos escogido.
    def __init__(self, opcion_dataset):
        self.opcion_dataset = opcion_dataset
     
    # Método interacciones_movielens. Crea las interacciones de movielens con las que poder utilizar los modelos.        
    def interacciones_movielens(self):
        global train
        global test       
    
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
        interacciones = Interactions(users_ids, items_ids, ratings, num_users=len(np.unique(users_ids)), num_items=len(np.unique(items_ids)))
        
        # Divido las interacciones en train y test
        train, test = random_train_test_split(interacciones)
        
    # Método interacciones_anime. Crea las interacciones de movielens con las que poder utilizar los modelos.        
    def interacciones_anime(self):
        global train
        global test       
    
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
        if self.opcion_dataset == 1:
            self.interacciones_movielens()
        elif self.opcion_dataset == 2:
            self.interacciones_anime()
        elif self.opcion_dataset == 3:
            self.interacciones_book_crossing()
        elif self.opcion_dataset == 4:
            self.interacciones_lastfm()
        else:
            self.interacciones_dating_agency()
            
    # Método modelo_factorizacion_explicito. Crea el modelo de factorización explícito.
    def modelo_factorizacion_explitico(self):
        global train
        global modelo
        
        # Obtengo y entreno el modelo
        modelo = ExplicitFactorizationModel(n_iter=1)
        modelo.fit(train)
            
    # Método obtener_modelos. Crea los modelos en función de la opción escogida.
    def obtener_modelos(self):
        self.modelo_factorizacion_explitico()
        
    # Método resultados_factorizacion_explicito. Calcula las métricas del modelo de factorización explícito.
    def resultados_factorizacion_explicito(self):
        global train
        global test
        global modelo
        
        # Calculo las métricas
        rmse = rmse_score(modelo, test)
        mrr = mrr_score(modelo, test, train=train).mean()
        precision, recall = precision_recall_score(modelo, test, train=train, k=10)
        
        Salida.imprimir_resultados_dl(rmse, mrr, precision.mean(), recall.mean())
        
    # Método obtener_resultados. Calcula las métricas en función del modelo escogido.
    def obtener_resultados(self):
        self.resultados_factorizacion_explicito()
    
    
    