# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:40:22 2019

@author: Raúl
"""


# Se importa todo lo necesario
import torch
import numpy as np
from modelo import Entrada
from modelo.Salida import imprimir_resultados_dl
from modelo.Persistencia import guardar_datos_pickle, guardar_modelos_dl, cargar_datos_pickle, cargar_modelo_dl, guardar_resultados
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score, mrr_score, precision_recall_score, sequence_mrr_score


class SistemaSpotlight:
    """
    Clase SistemaSpotlight.
    
    Parameters
    ----------
    
    opcion_modelo: int
        modelo que se quiere obtener.
    opcion_time: int, optional
        opcion por si se quiere utilizar timestamps.
        
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
    

    def __init__(self, opcion_modelo, opcion_time=None):        
        self.opcion_modelo = opcion_modelo
        if opcion_time is not None:
            self.opcion_time = opcion_time
           

    def obtener_interacciones(self):
        """
        Método obtener_interacciones. Obtiene las interacciones necesarias por los modelos de Spotlight.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, test
        
        # Se obtiene el dataframe de valoraciones
        Entrada.obtener_datos()
        ratings_df = Entrada.ratings_df

        # Se obtienen arrays con los ids de los usuarios y de los ítems
        users_ids = np.asarray(ratings_df[ratings_df.columns.values[0]].tolist(), dtype=np.int32)         
        items_ids = np.asarray(ratings_df[ratings_df.columns.values[1]].tolist(), dtype=np.int32)
        
        # Se transforma el dataframe de valoraciones en interacciones que puedan ser utilzadas por los modelos
        if self.opcion_time == 1:
            timestamps = np.asarray(ratings_df[ratings_df.columns.values[3]].tolist(), dtype=np.int32)
            if self.opcion_modelo == 1:
                ratings = np.asarray(ratings_df[ratings_df.columns.values[2]].tolist(), dtype=np.float32)
                interacciones = Interactions(users_ids, items_ids, ratings=ratings, timestamps=timestamps)
                train, test = random_train_test_split(interacciones)
            else:
                interacciones = Interactions(users_ids, items_ids, timestamps=timestamps)
                train, test = random_train_test_split(interacciones)
                if self.opcion_modelo == 3:
                    train = train.to_sequence()
                    test = test.to_sequence()
        else:
            if self.opcion_modelo == 1:
                ratings = np.asarray(ratings_df[ratings_df.columns.values[2]].tolist(), dtype=np.float32)
                interacciones = Interactions(users_ids, items_ids, ratings=ratings)
            else:
                interacciones = Interactions(users_ids, items_ids)
            train, test = random_train_test_split(interacciones)
            
        # Se guardan las interacciones de entrenamiento y test
        print("Guarda las interacciones de train")
        guardar_datos_pickle(train, 'las interacciones de entrenamiento')
        print("Guarda las interacciones de test")
        guardar_datos_pickle(test, 'las interacciones de test')


    def obtener_interacciones_gui(self, ruta_ratings, sep_ratings, encoding_ratings):
        """
        Método obtener_interacciones_gui. Obtiene las interacciones necesarias para la creación de los modelos de Spotlight.

        Este método solo se utiliza en la interfaz web.

        Parameters
        ----------

        ruta_ratings: str
            ruta del archivo que contiene las valoraciones.
        sep_ratings: str
            separador utilizado en el archivo de valoraiones.
        encoding_ratings: str
            encoding utilizado en el archivo de valoraciones.
        """

        global train, test
        
        # Se obtiene el dataframe de valoraciones
        ratings_df = Entrada.leer_csv(ruta_ratings, sep_ratings, encoding_ratings)
        ratings_df.sort_values([ratings_df.columns.values[0], ratings_df.columns.values[1]], inplace=True)

        # Se obtienen arrays con los ids de los usuarios y de los ítems
        users_ids = np.asarray(ratings_df[ratings_df.columns.values[0]].tolist(), dtype=np.int32)         
        items_ids = np.asarray(ratings_df[ratings_df.columns.values[1]].tolist(), dtype=np.int32)
        
        # Se transforma el dataframe de valoraciones en interacciones que puedan ser utilzadas por los modelos
        if self.opcion_time == 1:
            timestamps = np.asarray(ratings_df[ratings_df.columns.values[3]].tolist(), dtype=np.int32)
            if self.opcion_modelo == 1:
                ratings = np.asarray(ratings_df[ratings_df.columns.values[2]].tolist(), dtype=np.float32)
                interacciones = Interactions(users_ids, items_ids, ratings=ratings, timestamps=timestamps)
                train, test = random_train_test_split(interacciones)
            else:
                interacciones = Interactions(users_ids, items_ids, timestamps=timestamps)
                train, test = random_train_test_split(interacciones)
                if self.opcion_modelo == 3:
                    train = train.to_sequence()
                    test = test.to_sequence()
        else:
            if self.opcion_modelo == 1:
                ratings = np.asarray(ratings_df[ratings_df.columns.values[2]].tolist(), dtype=np.float32)
                interacciones = Interactions(users_ids, items_ids, ratings=ratings)
            else:
                interacciones = Interactions(users_ids, items_ids)
            train, test = random_train_test_split(interacciones)
            
        # Se guardan las interacciones de entrenamiento y test
        print("Guarda las interacciones de train")
        guardar_datos_pickle(train, 'las interacciones de entrenamiento')
        print("Guarda las interacciones de test")
        guardar_datos_pickle(test, 'las interacciones de test')


    def cargar_interacciones_gui(self, ruta_train, ruta_test):
        """
        Método cargar_interacciones_gui. Carga las interacciones necesarias para la creación de los modelos de Spotlight.

        Este método solo se utiliza en la interfaz web.

        Parameters
        ----------

        ruta_train: str
            ruta del archivo que contiene el conjunto de entrenamiento.
        ruta_test: str
            ruta del archivo que contiene el conjunto de test.
        """

        global train, test

        # Se cargan las interacciones
        train = cargar_datos_pickle(ruta_train)
        test = cargar_datos_pickle(ruta_test)


    def cargar_otras_interacciones_gui(self):
        """
        Método cargar_otras_interacciones_gui. Carga las interacciones de nuevos datasets necesarias para la creación de los modelos de Spotlight.

        Este método solo se utiliza en la interfaz web.
        """

        global train, test

        # Se pregunta dónde están los archivos y se cargan
        ruta_train = Entrada.elegir_archivo('entrenamiento')
        train = cargar_datos_pickle(ruta_train)
        ruta_test = Entrada.elegir_archivo('test')
        test = cargar_datos_pickle(ruta_train)


    def obtener_modelos(self):
        """
        Método obtener_modelos. Obtiene, entrena y guarda el modelo escogido.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, modelo
        
        # Se obtiene el modelo, se entrena con parámetros por defecto y se guarda
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
        """
        Método obtener_modelo_gui. Obtiene el modelo escogido según los parámetros pasados.

        Este método solo se utiliza en la interfaz web.

        Parameters
        ----------

        lista_param: list
            lista que contiene los parámetros escogidos por el usuario para crear el modelo.
        """

        global modelo

        # Se guardan los parámetros en variables para que sea más legible
        loss = lista_param[0]
        embedding_dim = lista_param[1]
        n_iter = lista_param[2]
        batch_size = lista_param[3]
        l2 = lista_param[4]
        learning_rate = lista_param[5]
        representation = lista_param[6]

        # Se instancia el modelo según los parámetros anteriores
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
        """
        Método cargar_modelo_gui. Carga el modelo escogido.

        Este método solo se utiliza en la interfaz web.

        Parameters
        ----------

        ruta_modelo: str
            ruta del archivo que contiene el modelo escogido.
        """

        global modelo

        # Se carga el modelo escogido
        modelo = cargar_modelo_dl(ruta_modelo)


    def entrenar_modelo_gui(self):
        """
        Método entrenar_modelo_gui. Entrena el modelo escogido.

        Este método solo se utiliza en la interfaz web.
        """

        global modelo, train

        # Se entrena el modelo y se guarda
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

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, test, modelo
        
        # Se calculan las métricas
        rmse = rmse_score(modelo, test)
        mrr = mrr_score(modelo, test, train=train).mean()
        precision, recall = precision_recall_score(modelo, test, train=train, k=10)
        
        # Se imprimen las métricas
        imprimir_resultados_dl(mrr, precision.mean(), recall.mean(), rmse)
        

    def resultados_factorizacion_implicito(self):
        """
        Método resultados_factorizacion_implicito. Calcula las métricas del modelo de factorización implícito.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, test, modelo
        
        # Se calculan las métricas
        mrr = mrr_score(modelo, test, train=train).mean()
        precision, recall = precision_recall_score(modelo, test, train=train, k=10)
        
        # Se imprimen las métricas
        imprimir_resultados_dl(mrr, precision.mean(), recall.mean())
        

    def resultados_secuencia(self):
        """
        Método resultados_secuencia. Calcula las métricas del modelo de secuencia implícito.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, test, modelo
        
        # Se calculan las métricas
        mrr = sequence_mrr_score(modelo, test).mean()
        #precision, recall = sequence_precision_recall_score(modelo, test)
        
        # Se imprimen las métricas
        imprimir_resultados_dl(mrr)
        

    def obtener_resultados(self):
        """
        Método obtener_resultados. Calcula las métricas en función del modelo escogido.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        if self.opcion_modelo == 1:
            self.resultados_factorizacion_explicito()
        elif self.opcion_modelo == 2:
            self.resultados_factorizacion_implicito()
        else:
            self.resultados_secuencia()
    

    def obtener_metricas_gui(self):
        """
        Método obtener_metricas_gui. Obtiene las métricas del modelo escogido.

        Este método solo se utiliza en la interfaz web.
        """

        global train, test, modelo

        # Se guardan las métricas en un diccionario para su futura muestra en la interfaz web
        metricas = dict()

        # Se calculan las métricas y se guardan en el diccionario formateadas
        if self.opcion_modelo == 1:
            rmse = rmse_score(modelo, test)
            mrr = mrr_score(modelo, test, train=train).mean()
            precision, recall = precision_recall_score(modelo, test, train=train, k=10)
            metricas_devueltas = {"RMSE": format(rmse, '.4f'), "MRR": format(mrr, '.4f'), "Precisión k": format(precision.mean(), '.4f'), "Recall k": format(recall.mean(), '.4f')}
            metricas_a_guardar = {"RMSE": [format(rmse, '.4f')], "MRR": [format(mrr, '.4f')], "Precisión k": [format(precision.mean(), '.4f')], "Recall k": [format(recall.mean(), '.4f')]}
        elif self.opcion_modelo == 2:
            mrr = mrr_score(modelo, test, train=train).mean()
            precision, recall = precision_recall_score(modelo, test, train=train, k=10)
            metricas_devueltas = {"MRR": format(mrr, '.4f'), "Precisión k": format(precision.mean(), '.4f'), "Recall k": format(recall.mean(), '.4f')}
            metricas_a_guardar = {"MRR": [format(mrr, '.4f')], "Precisión k": [format(precision.mean(), '.4f')], "Recall k": [format(recall.mean(), '.4f')]}
        else:
            mrr = sequence_mrr_score(modelo, test).mean()
            metricas_devueltas = {"MRR": format(mrr, '.4f')}
            metricas_a_guardar = {"MRR": [format(mrr, '.4f')]}
        
        # Se guardan las métricas en un archivo .csv
        guardar_resultados(metricas_a_guardar)

        return metricas_devueltas


    def obtener_datos_conjunto_gui(self):
        """
        Método obtener_datos_conjunto_gui. Obtiene los datos del dataset escogido.

        Este método solo se utiliza en la interfaz web.
        """
        
        global train, test

        # Se guardan los datos en un diccionario para su futura muestra en la interfaz web
        datos = dict()

        # Se guarda el número de usuarios, de ítems y de valoraciones
        if self.opcion_modelo == 1 or self.opcion_modelo == 2:
            datos = {"Usuarios": train.tocoo().shape[0]-1, "Items": train.tocoo().shape[1]-1, "Valoraciones": train.tocoo().nnz+test.tocoo().nnz}
        else:
            datos = None

        return datos


    def obtener_id_maximo(self):
        """
        Método obtener_id_maximo. Obtiene el id del último usuario del dataset escogido.
        Se utiliza para indicarle al usuario de la aplicación que no debe pasarse de ese número.

        Este método solo se utiliza en la interfaz web.
        """

        global train

        return train.tocoo().shape[0]
    
    
    def obtener_predicciones(self, usuario):
        """
        Método obtener_predicciones. Obtiene las 20 primeras predicciones para el usuario escogido.

        Este método solo se utiliza en la interfaz web.

        Parameters
        ----------

        usuario: int
            id del usuario cuyas predicciones se quieren obtener
        """

        global modelo

        scores = modelo.predict(usuario)

        # Se obtienen los ids de los ítems ordenando las predicciones de mayor a menor en función del score obtenido
        predicciones = np.argsort(-scores)

        # Se devuelven las 20 mejores predicciones
        return predicciones[:20]


