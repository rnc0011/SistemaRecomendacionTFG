# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:40:22 2019

@author: Raúl
"""

# Importo todo lo necesario
import torch
import numpy as np
from modelo import Entrada
from modelo.Salida import imprimir_resultados_dl
from modelo.Persistencia import guardar_datos_pickle, guardar_modelos_dl, cargar_datos_pickle, cargar_modelo_dl
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score, mrr_score, precision_recall_score, sequence_mrr_score
#from spotlight.evaluation import rmse_score, mrr_score, precision_recall_score, sequence_mrr_score, sequence_precision_recall_score

class SistemaSpotlight:
    """
    Clase SistemaSpotlight.
    
    Parameters
    ----------
    
    opcion_modelo: int
        modelo que se quiere obtener.
    opcion_time: int
        opcion por si se quiere utilizar timestamps
        
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
    
    def __init__(self, opcion_modelo=None, opcion_time=None):        
        if opcion_modelo is not None:
            self.opcion_modelo = opcion_modelo
        if opcion_time is not None:
            self.opcion_time = opcion_time
           

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
        
        # Obtengo las interacciones
        if self.opcion_time == 1:
            timestamps = np.asarray(ratings_df[ratings_df.columns.values[3]].tolist(), dtype=np.int32)
            if self.opcion_modelo == 1:
                ratings = np.asarray(ratings_df[ratings_df.columns.values[2]].tolist(), dtype=np.float32)
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
        else:
            if self.opcion_modelo == 1:
                ratings = np.asarray(ratings_df[ratings_df.columns.values[2]].tolist(), dtype=np.float32)
                interacciones = Interactions(users_ids, items_ids, ratings=ratings, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
                train, test = random_train_test_split(interacciones)
            else:
                interacciones = Interactions(users_ids, items_ids, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
                train, test = random_train_test_split(interacciones)
            
        print("Guarda las interacciones de train")
        guardar_datos_pickle(train, 'las interacciones de entrenamiento')
        print("Guarda las interacciones de test")
        guardar_datos_pickle(test, 'las interacciones de test')


    def obtener_interacciones_gui(self, ratings_df):
        global train, test
        
        users_ids = np.asarray(ratings_df[ratings_df.columns.values[0]].tolist(), dtype=np.int32)         
        items_ids = np.asarray(ratings_df[ratings_df.columns.values[1]].tolist(), dtype=np.int32)
        
        # Obtengo las interacciones
        if self.opcion_time == 1:
            timestamps = np.asarray(ratings_df[ratings_df.columns.values[3]].tolist(), dtype=np.int32)
            if self.opcion_modelo == 1:
                ratings = np.asarray(ratings_df[ratings_df.columns.values[2]].tolist(), dtype=np.float32)
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
        else:
            if self.opcion_modelo == 1:
                ratings = np.asarray(ratings_df[ratings_df.columns.values[2]].tolist(), dtype=np.float32)
                interacciones = Interactions(users_ids, items_ids, ratings=ratings, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
                train, test = random_train_test_split(interacciones)
            else:
                interacciones = Interactions(users_ids, items_ids, num_users=len(np.unique(users_ids))+1, num_items=len(np.unique(items_ids))+1)
                train, test = random_train_test_split(interacciones)
            
        print("Guarda las interacciones de train")
        guardar_datos_pickle(train, 'las interacciones de entrenamiento')
        print("Guarda las interacciones de test")
        guardar_datos_pickle(test, 'las interacciones de test')


    def cargar_interacciones_gui(self, ruta_train, ruta_test):
        global train, test

        train = cargar_datos_pickle(ruta_train)
        test = cargar_datos_pickle(ruta_test)


    def cargar_otras_interacciones_gui(self):
        global train, test

        ruta_train = Entrada.elegir_archivo('entrenamiento')
        train = cargar_datos_pickle(ruta_train)
        ruta_test = Entrada.elegir_archivo('test')
        test = cargar_datos_pickle(ruta_train)


    def obtener_modelos(self):
        """
        Método obtener_modelos. Obtiene el modelo escogido.
        """
        
        global train, modelo
        
        # Obtengo entreno y guardo el modelo
        if self.opcion_modelo == 1:
            modelo = ExplicitFactorizationModel(loss='logistic', use_cuda=torch.cuda.is_available())
            modelo.fit(train, verbose=True)
            guardar_modelos_dl(modelo, 'el modelo de factorización explícito')
        elif self.opcion_modelo == 2:
            modelo = ImplicitFactorizationModel(loss='bpr', use_cuda=torch.cuda.is_available())
            modelo.fit(train, verbose=True)
            guardar_modelos_dl(modelo, 'el modelo de factorización implícito')
        else:
            modelo = ImplicitSequenceModel(loss='bpr',  representation='pooling', use_cuda=torch.cuda.is_available())
            modelo.fit(train, verbose=True)
            guardar_modelos_dl(modelo, 'el modelo de secuencia explícito')


    def obtener_modelo_gui(self, lista_param):
        global modelo

        loss = lista_param[0]
        embedding_dim = lista_param[1]
        n_iter = lista_param[2]
        batch_size = lista_param[3]
        l2 = lista_param[4]
        learning_rate = lista_param[5]
        representation = lista_param[6]

        if self.opcion_modelo == 1:
            modelo = ExplicitFactorizationModel(loss=loss, embedding_dim=embedding_dim, n_iter=n_iter, batch_size=batch_size, 
                l2=l2, learning_rate=learning_rate, use_cuda=torch.cuda.is_available())
        elif self.opcion_modelo == 2:
            modelo = ImplicitFactorizationModel(loss=loss, embedding_dim=embedding_dim, n_iter=n_iter, batch_size=batch_size, 
                l2=l2, learning_rate=learning_rate, use_cuda=torch.cuda.is_available())
        else:
            modelo = ImplicitSequenceModel(loss=loss, representation=representation, embedding_dim=embedding_dim, n_iter=n_iter, batch_size=batch_size, 
                l2=l2, learning_rate=learning_rate, use_cuda=torch.cuda.is_available())


    def cargar_modelo_gui(self, ruta_modelo):
        global modelo

        modelo = cargar_modelo_dl(ruta_modelo)


    def entrenar_modelo_gui(self):
        global modelo, train

        if self.opcion_modelo == 1:
            modelo.fit(train, verbose=True)
            guardar_modelos_dl(modelo, 'el modelo de factorización explícito')
        elif self.opcion_modelo == 2:
            modelo.fit(train, verbose=True)
            guardar_modelos_dl(modelo, 'el modelo de factorización implícito')
        else:
            modelo.fit(train, verbose=True)
            guardar_modelos_dl(modelo, 'el modelo de secuencia explícito')
        

    def resultados_factorizacion_explicito(self):
        """
        Método resultados_factorizacion_explicito. Calcula las métricas del modelo de factorización explícito.
        """
        
        global train, test, modelo
        
        # Calculo las métricas
        rmse = rmse_score(modelo, test)
        mrr = mrr_score(modelo, test, train=train).mean()
        precision, recall = precision_recall_score(modelo, test, train=train, k=10)
        
        imprimir_resultados_dl(mrr, precision.mean(), recall.mean(), rmse)
        

    def resultados_factorizacion_implicito(self):
        """
        Método resultados_factorizacion_implicito. Calcula las métricas del modelo de factorización implícito.
        """
        
        global train, test, modelo
        
        # Calculo las métricas
        mrr = mrr_score(modelo, test, train=train).mean()
        precision, recall = precision_recall_score(modelo, test, train=train, k=10)
        
        imprimir_resultados_dl(mrr, precision.mean(), recall.mean())
        

    def resultados_secuencia(self):
        """
        Método resultados_secuencia. Calcula las métricas del modelo de secuencia implícito.
        """
        
        global train, test, modelo
        
        mrr = sequence_mrr_score(modelo, test).mean()
        #precision, recall = sequence_precision_recall_score(modelo, test)
        
        imprimir_resultados_dl(mrr)
        

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
    

    def obtener_metricas_gui(self):
        global train, test, modelo

        metricas = dict()
        if self.opcion_modelo == 1:
            rmse = rmse_score(modelo, test)
            mrr = mrr_score(modelo, test, train=train).mean()
            precision, recall = precision_recall_score(modelo, test, train=train, k=10)
            metricas = {"RMSE": format(rmse, '.4f'), "MRR": format(mrr, '.4f'), "Precisión k": format(precision.mean(), '.4f'), "Recall k": format(recall.mean(), '.4f')}
        elif self.opcion_modelo == 2:
            mrr = mrr_score(modelo, test, train=train).mean()
            precision, recall = precision_recall_score(modelo, test, train=train, k=10)
            metricas = {"MRR": format(mrr, '.4f'), "Precisión k": format(precision.mean(), '.4f'), "Recall k": format(recall.mean(), '.4f')}
        else:
            mrr = sequence_mrr_score(modelo, test).mean()
            metricas = {"MRR": format(mrr, '.4f')}
        return metricas


    def obtener_datos_conjunto_gui(self):
        global train, test

        datos = dict()
        if self.opcion_modelo == 1 or self.opcion_modelo == 2:
            datos = {"Usuarios": train.tocoo().shape[0]-1, "Items": train.tocoo().shape[1]-1, "Valoraciones": train.tocoo().nnz+test.tocoo().nnz}
        else:
            datos = None
        return datos


    def obtener_id_maximo(self):
        global train

        return train.tocoo().shape[0] - 1
    
    
    def obtener_predicciones(self, usuario):
        global modelo

        scores = modelo.predict(usuario)
        """if self.opcion_modelo == 1 or self.opcion_modelo == 2:
            scores = modelo.predict(usuario)
        else:
            # no es usuario, debería ser secuencia
            scores = modelo.predict([usuario])"""
        predicciones = np.argsort(-scores)
        return predicciones[:20]


