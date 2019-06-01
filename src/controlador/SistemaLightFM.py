# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:45:27 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import multiprocessing
#import pickle
#import os
import sys
sys.path.insert(0, 'C:\\Users\\Raúl\\Google Drive\\GitHub\\SistemaRecomendacionTFG\\src\\modelo')
import Entrada
import Salida
import Persistencia
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score, recall_at_k, reciprocal_rank
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split

# Clase SistemaLightFM.
class SistemaLightFM:
    
    # Variables globales
    global train, test, modelo, item_features, user_features

    # Constantes
    CPU_THREADS = multiprocessing.cpu_count()
    
    # Método __init__. Inicializa la clase con el conjunto de datos y el modelo escogidos.
    def __init__(self, opcion_modelo):
        self.opcion_modelo = opcion_modelo
    
    def obtener_matrices(self):
        Entrada.obtener_datos()
        # Resto de cosas
    
    # Método modelo_colaborativo. Crea el modelo colaborativo.
    def modelo_colaborativo(self):
        global train, modelo
        
        # Obtención del modelo
        modelo = LightFM(loss='warp')
        modelo.fit(train, epochs=30, num_threads=self.CPU_THREADS)
        
        # Guardo el modelo
        Persistencia.guardar_modelo_clasico(modelo, 'colaborativo', self.opcion_dataset)
        
    # Método modelo_hibrido. Crea el modelo híbrido.
    def modelo_hibrido(self):
        global train, modelo, item_features
        
        # Obtención del modelo
        modelo = LightFM(loss='warp')
        modelo.fit(train, item_features=item_features, epochs=30, num_threads=self.CPU_THREADS)
        
        # Guardo el modelo
        Persistencia.guardar_modelo_clasico(modelo, 'hibrido', self.opcion_dataset)
        
    # Método modelo_por_contenido. Crea el modelo por contenido.
    def modelo_por_contenido(self):
        global train, modelo, item_features, user_features
        
        # Obtención del modelo
        modelo = LightFM(loss='warp')
        modelo.fit(train, user_features=user_features, item_features=item_features, epochs=30, num_threads=self.CPU_THREADS)
        
        # Guardo el modelo
        Persistencia.guardar_modelo_clasico(modelo, 'contenido', self.opcion_dataset)
    
    # Método obtener_modelo. Crea los modelos en función de la opción escogida.
    def obtener_modelos(self):
        if self.opcion_modelo == 1:
            self.modelo_colaborativo()
        elif self.opcion_modelo == 2:
            self.modelo_hibrido()
        else:
            self.modelo_por_contenido()
            
    # Método resultados_colaboraivo. Obtiene los resultados del modelo colaborativo.
    def resultados_colaborativo(self):
        global train, test, modelo
        
        # Obtención de los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()
        
        Salida.imprimir_resultados_clasico(precision, auc, recall, reciprocal)
        
    # Método resultados_hibrido. Obtiene los resultados del modelo híbrido.
    def resultados_hibrido(self):
        global train, test, modelo, item_features
        
        # Obtención de los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        
        Salida.imprimir_resultados_clasico(precision, auc, recall, reciprocal)
    
    # Método resultados_por_contenido. Obtiene los resultados del modelo por contenido.
    def resultados_por_contenido(self):
        global train, test, modelo, item_features, user_features
        
        # Obtención de los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        
        Salida.imprimir_resultados_clasico(precision, auc, recall, reciprocal)
    
    # Método obtener_resultados. Obtiene los resultados en función del modelo escogido.
    def obtener_resultados(self):
        if self.opcion_modelo == 1:
            self.resultados_colaborativo()
        elif self.opcion_modelo == 2:
            self.resultados_hibrido()
        else:
            self.resultados_por_contenido()
    
    
    
    
    