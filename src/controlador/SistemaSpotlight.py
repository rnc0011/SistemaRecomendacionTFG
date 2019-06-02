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
    
    Parameters
    ----------
    
    opcion_modelo: int
        modelo que se quiere obtener.
        
    Attributes
    ----------
    
    train: Interactions
        conjunto de entranamiento.
    test: Interactions
        conjunto de test.
    modelo: ImplicitFactorizationModel, ExplicitFactorizationModelo or ImplicitSequenceModel instance
        modelo a evaluar.
    """
    
    # Variables globales
    global train, test, modelo
    
    def __init__(self, opcion_modelo):        
        self.opcion_modelo = opcion_modelo
           
    def obtener_interacciones(self):
        """
        Método obtener_interacciones. Obtiene las interacciones necesarias por los modelos.
        """
        
        global train, test
        
        # Obtengo los datos
        Entrada.obtener_datos()
        ratings_df = Entrada.ratings_df
        users_ids = np.asarray(ratings_df[ratings_df.columns.values[0]].tolist(), dtype=np.int32)         
        items_ids = np.asarray(ratings_df[ratings_df.columns.values[1]].tolist(), dtype=np.int32)
        #timestamps = np.asarray(ratings_df['Fecha'].tolist(), dtype=np.int32)
        
        # Obtengo las interacciones
        if self.opcion_modelo == 1:
            ratings = np.asarray(ratings_df[ratings_df.columns.values[2]].tolist(), dtype=np.float32)
            interacciones = Interactions(users_ids, items_ids, ratings=ratings, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
            train, test = random_train_test_split(interacciones)
        elif self.opcion_modelo == 2:
            interacciones = Interactions(users_ids, items_ids, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
            train, test = random_train_test_split(interacciones)
        else:
            interacciones = Interactions(users_ids, items_ids, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
            train, test = random_train_test_split(interacciones)
            train = train.to_sequence()
            test = test.to_sequence()

    def obtener_modelos(self):
        """
        Método obtener_modelos. Obtiene el modelo escogido.
        """
        
        global train, modelo
        
        if self.opcion_modelo == 1:
            modelo = ExplicitFactorizationModel(loss='logistic', use_cuda=True)
        elif self.opcion_modelo == 2:
            modelo = ImplicitFactorizationModel(loss='bpr', use_cuda=True)
        else:
            modelo = ImplicitSequenceModel(loss='bpr',  representation='pooling', use_cuda=True)
            
        modelo.fit(train, verbose=True)
        
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
    
    
    