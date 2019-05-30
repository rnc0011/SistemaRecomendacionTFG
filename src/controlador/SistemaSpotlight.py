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
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score, mrr_score, precision_recall_score, sequence_mrr_score#, sequence_precision_recall_score

class SistemaSpotlight:
    """
    Clase SistemaSpotlight.
    """
    
    # Variables globales
    global train, test, modelo
    
    def __init__(self, opcion_dataset, opcion_modelo):        
        self.opcion_dataset = opcion_dataset
        self.opcion_modelo = opcion_modelo
           
    def interacciones_movielens(self):
        """
        Método interacciones_movielens. Crea las interacciones de movielens con las que poder utilizar los modelos.
        """
        
        global train, test
    
        # Leo los csv
        Entrada.leer_movielens()
        
        # Obtengo los arrays con los ids de los usuarios, los items y los timestamps
        users_ids = np.asarray(Entrada.ratings_df['Id Usuario'].tolist(), dtype=np.int32)         
        items_ids = np.asarray(Entrada.ratings_df['Id Película'].tolist(), dtype=np.int32)
        timestamps = np.asarray(Entrada.ratings_df['Fecha'].tolist(), dtype=np.int32)
                
        # Obtengo el array con las valoraciones    
        if self.opcion_modelo == 1:
            ratings = np.asarray(Entrada.ratings_df['Valoración'].tolist(), dtype=np.float32)
            interacciones = Interactions(users_ids, items_ids, ratings=ratings, timestamps=timestamps, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
            train, test = random_train_test_split(interacciones)
        elif self.opcion_modelo == 2:
            interacciones = Interactions(users_ids, items_ids, timestamps=timestamps, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
            train, test = random_train_test_split(interacciones)
        else:
            interacciones = Interactions(users_ids, items_ids, timestamps=timestamps, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
            train, test = random_train_test_split(interacciones)
            train = train.to_sequence()
            test = test.to_sequence()
              
    def interacciones_anime(self):
        """
        Método interacciones_anime. Crea las interacciones de movielens con las que poder utilizar los modelos.  
        """
        
        global train, test       
    
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
               
    def interacciones_book_crossing(self):
        """
        Método interacciones_book_crossing. Crea las interacciones de movielens con las que poder utilizar los modelos. 
        """
        
        global train, test       
    
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
               
    def interacciones_lastfm(self):
        """
        Método interacciones_lastfm. Crea las interacciones de movielens con las que poder utilizar los modelos. 
        """
        
        global train, test       
    
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
              
    def interacciones_dating_agency(self):
        """
        Método interacciones_dating_agency. Crea las interacciones de movielens con las que poder utilizar los modelos.  
        """
        
        global train, test    
    
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
           
    def obtener_interacciones(self):
        """
        Método obtener_interacciones. Crea las interacciones con las que poder utilizar los modelos en función del dataset escogido. 
        """
        
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
            
    def modelo_factorizacion_explicito(self):
        """
        Método modelo_factorizacion_explicito. Crea el modelo de factorización explícito.
        """
        
        global train, modelo
        
        # Obtengo y entreno el modelo
        modelo = ExplicitFactorizationModel(loss='logistic', use_cuda=True)
        modelo.fit(train, verbose=True)
        
    def modelo_factorizacion_implicito(self):
        """
        Método modelo_factorizacion_implicito. Crea el modelo de factorización implícito.
        """
        
        global train, modelo
        
        # Obtengo y entreno el modelo
        modelo = ImplicitFactorizationModel(loss='bpr', use_cuda=True)
        modelo.fit(train, verbose=True)
        
    def modelo_secuencia(self):
        """
        Método modelo_secuencia. Crea el modelo de secuencia implícito. 
        """
        
        global train, modelo
        
        # Obtengo y entreno el modelo
        modelo = ImplicitSequenceModel(loss='bpr',  representation='pooling', use_cuda=True)
        modelo.fit(train, verbose=True)
        
    """
    def modelos(self):
        global train, modelo
        
        if self.opcion_modelo == 1:
            modelo = ExplicitFactorizationModel(loss='logistic', use_cuda=True)
        elif self.opcion_modelo == 2:
            modelo = ImplicitFactorizationModel(loss='bpr', use_cuda=True)
        else:
            modelo = ImplicitSequenceModel()
            
        modelo.fit(train, verbose=True)
    """
            
    def obtener_modelos(self):
        """
        Método obtener_modelos. Crea los modelos en función de la opción escogida.
        """
        
        if self.opcion_modelo == 1:
            self.modelo_factorizacion_explicito()
        elif self.opcion_modelo == 2:
            self.modelo_factorizacion_implicito()
        else:
            self.modelo_secuencia()
        
    def resultados_factorizacion_explicito(self):
        """
        Método resultados_factorizacion_explicito. Calcula las métricas del modelo de factorización explícito.
        """
        
        global train, test, modelo
        
        # Calculo las métricas
        rmse = rmse_score(modelo, test)
        mrr = mrr_score(modelo, test, train=train).mean()
        precision, recall = precision_recall_score(modelo, test, train=train, k=10)
        
        Salida.imprimir_resultados_dl(mrr, precision.mean(), recall.mean(), rmse)
        
    def resultados_factorizacion_implicito(self):
        """
        Método resultados_factorizacion_implicito. Calcula las métricas del modelo de factorización implícito.
        """
        
        global train, test, modelo
        
        # Calculo las métricas
        mrr = mrr_score(modelo, test, train=train).mean()
        precision, recall = precision_recall_score(modelo, test, train=train, k=10)
        
        Salida.imprimir_resultados_dl(mrr, precision.mean(), recall.mean())
        
    def resultados_secuencia(self):
        """
        Método resultados_secuencia. Calcula las métricas del modelo de secuencia implícito.
        """
        
        global train, test, modelo
        
        mrr = sequence_mrr_score(modelo, test).mean()
        #precision, recall = sequence_precision_recall_score(modelo, test)
        
        Salida.imprimir_resultados_dl(mrr)
        
    def obtener_resultados(self):
        """
        Método obtener_resultados. Calcula las métricas en función del modelo escogido.
        """
        
        if self.opcion_modelo == 1:
            self.resultados_factorizacion_explicito()
        elif self.opcion_modelo == 2:
            self.resultados_factorizacion_implicito()
        else:
            self.resultados_secuencia()
    
    
    