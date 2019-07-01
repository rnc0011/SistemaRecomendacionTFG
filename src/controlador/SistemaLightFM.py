# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:45:27 2019

@author: Raúl Negro Carpintero
"""


# Se importa todo lo necesario
import multiprocessing
import numpy as np
from modelo import Entrada
from modelo.Salida import imprimir_resultados_clasico
from modelo.Persistencia import guardar_datos_pickle, cargar_datos_pickle
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score, recall_at_k, reciprocal_rank
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split


class SistemaLightFM:
    """
    Clase SistemaLightFM.
    
    Parameters
    ----------
    
    opcion_modelo: int
        modelo que se quiere obtener.
    epochs: int, optional
        epochs que se quieren utilzar durante el entrenamiento del modelo.
        
    Attributes
    ----------
    
    train: np.float32 csr_matrix of shape [n_users, n_items]
        conjunto de entranamiento.
    test: np.float32 csr_matrix of shape [n_users, n_items]
        conjunto de test.
    modelo: LightFM instance
        modelo a evaluar.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features]
        características de los items.
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features]
        características de los usuarios.
    """
    

    # Constantes
    CPU_THREADS = multiprocessing.cpu_count()
    

    # Variables globales
    global train, test, modelo, item_features, user_features
    

    def __init__(self, opcion_modelo, epochs=None):
        self.opcion_modelo = opcion_modelo
        if epochs is not None:
            self.epochs = epochs
  

    def obtener_matrices(self):
        """
        Método obtener_matrices. Obtiene las matrices necesarias para la creación de los modelos de LightFM.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, test, modelo, item_features, user_features
        
        # Se obtienen los dataframes
        Entrada.obtener_datos()
        ratings_df = Entrada.ratings_df
        users_df = Entrada.users_df
        items_df = Entrada.items_df

        # Se transforman los dataframes en matrices que puedan ser utilzadas por los modelos
        dataset = Dataset()
        dataset.fit(users_df[users_df.columns.values[0]], items_df[items_df.columns.values[0]],
                       user_features=users_df[users_df.columns.values[1]], item_features=items_df[items_df.columns.values[1]])
        
        # Si el modelo es colaborativo o híbrido se tienen en cuenta las valoraciones de los usuarios
        if self.opcion_modelo == 1 or self.opcion_modelo == 2:
            (interacciones, pesos) = dataset.build_interactions((row[ratings_df.columns.values[0]],
                                                                 row[ratings_df.columns.values[1]],
                                                                 row[ratings_df.columns.values[2]]) 
                                                                for index,row in ratings_df.iterrows())
        else:
            (interacciones, pesos) = dataset.build_interactions((row[ratings_df.columns.values[0]],
                                                                 row[ratings_df.columns.values[1]]) 
                                                                for index,row in ratings_df.iterrows())

        # Se obtienen las matrices de features y se guardan
        item_features = dataset.build_item_features((row[items_df.columns.values[0]], [row[items_df.columns.values[1]]]) for index, row in items_df.iterrows())
        user_features = dataset.build_user_features((row[users_df.columns.values[0]], [row[users_df.columns.values[1]]]) for index, row in users_df.iterrows())
        print("Guarda la matriz de item features")
        guardar_datos_pickle(item_features, 'la matriz de item features')
        print("Guarda la matriz de user features")
        guardar_datos_pickle(user_features, 'la matriz de user feautures')

        # Se dividen las interacciones en conjuntos de entrenamiento y test y se guardan
        train, test = random_train_test_split(interacciones, test_percentage=0.2)
        print("Guarda la matriz de entrenamiento")
        guardar_datos_pickle(train, 'la matriz de entrenamiento')
        print("Guarda la matriz de test")
        guardar_datos_pickle(test, 'la matriz de test')
          

    def obtener_matrices_gui(self, ruta_ratings, sep_ratings, encoding_ratings, ruta_users, sep_users, encoding_users, ruta_items, sep_items, encoding_items):
        """
        Método obtener_matrices_gui. Obtiene las matrices necesarias para la creación de los modelos de LightFM.

        Este método solo se utiliza en la interfaz web.

        Parameters
        ----------

        ruta_ratings: str
            ruta del archivo que contiene las valoraciones.
        sep_ratings: str
            separador utilizado en el archivo de valoraiones.
        encoding_ratings: str
            encoding utilizado en el archivo de valoraciones.
        ruta_users: str
            ruta del archivo que contiene los datos de los usuarios.
        sep_users: str
            separador utilizado en el archivo de usuarios.
        encoding_users: str
            encoding utilizado en el archivo de usuarios.
        ruta_items: str
            ruta del archivo que contiene los datos de los ítems.
        sep_items: str
            separador utilizado en el archivo de ítems.
        encoding_items: str
            encoding utilizado en el archivo de ítems.
        """

        global train, test, item_features, user_features

        # Se obtienen los dataframes
        ratings_df = Entrada.leer_csv(ruta_ratings, sep_ratings, encoding_ratings)
        users_df = Entrada.leer_csv(ruta_users, sep_users, encoding_users)
        items_df = Entrada.leer_csv(ruta_items, sep_items, encoding_items)

        # Se transforman los dataframes en matrices que puedan ser utilzadas por los modelos
        dataset = Dataset()
        dataset.fit(users_df[users_df.columns.values[0]], items_df[items_df.columns.values[0]],
                       user_features=users_df[users_df.columns.values[1]], item_features=items_df[items_df.columns.values[1]])
                
        # Si el modelo es colaborativo o híbrido se tienen en cuenta las valoraciones de los usuarios
        if self.opcion_modelo == 1 or self.opcion_modelo == 2:
            (interacciones, pesos) = dataset.build_interactions((row[ratings_df.columns.values[0]],
                                                                 row[ratings_df.columns.values[1]],
                                                                 row[ratings_df.columns.values[2]]) 
                                                                for index,row in ratings_df.iterrows())
        else:
            (interacciones, pesos) = dataset.build_interactions((row[ratings_df.columns.values[0]],
                                                                 row[ratings_df.columns.values[1]]) 
                                                                for index,row in ratings_df.iterrows())

        # Se obtienen las matrices de features y se guardan
        item_features = dataset.build_item_features((row[items_df.columns.values[0]], [row[items_df.columns.values[1]]]) for index, row in items_df.iterrows())
        user_features = dataset.build_user_features((row[users_df.columns.values[0]], [row[users_df.columns.values[1]]]) for index, row in users_df.iterrows())
        print("Guarda la matriz de item features")
        guardar_datos_pickle(item_features, 'la matriz de item features')
        print("Guarda la matriz de user features")
        guardar_datos_pickle(user_features, 'la matriz de user feautures')

        # Se dividen las interacciones en conjuntos de entrenamiento y test y se guardan
        train, test = random_train_test_split(interacciones, test_percentage=0.2)
        print("Guarda la matriz de entrenamiento")
        guardar_datos_pickle(train, 'la matriz de entrenamiento')
        print("Guarda la matriz de test")
        guardar_datos_pickle(test, 'la matriz de test')


    def cargar_matrices_gui(self, ruta_train, ruta_test, ruta_items, ruta_users):
        """
        Método cargar_matrices_gui. Carga las matrices de los datasets de prueba necesarias para la creación de los modelos de LightFM.

        Este método solo se utiliza en la interfaz web.

        Parameters
        ----------

        ruta_train: str
            ruta del archivo que contiene el conjunto de entrenamiento.
        ruta_test: str
            ruta del archivo que contiene el conjunto de test.
        ruta_items: str
            ruta del archivo que contiene las features de los ítems.
        ruta_users: str
            ruta del archivo que contiene las features de los usuarios.
        """

        global train, test, item_features, user_features

        # Se cargan las matrices
        train = cargar_datos_pickle(ruta_train)
        test = cargar_datos_pickle(ruta_test)
        item_features = cargar_datos_pickle(ruta_items)
        user_features = cargar_datos_pickle(ruta_users)


    def cargar_otras_matrices_gui(self):
        """
        Método cargar_otras_matrices_gui. Carga las matrices de nuevos datasets necesarias para la creación de los modelos de LightFM.

        Este método solo se utiliza en la interfaz web.
        """

        global train, test, item_features, user_features

        # Se pregunta dónde están los archivos y se cargan
        ruta_train = Entrada.elegir_archivo('entrenamiento')
        train = cargar_datos_pickle(ruta_train)
        ruta_test = Entrada.elegir_archivo('test')
        test = cargar_datos_pickle(ruta_test)
        ruta_items = Entrada.elegir_archivo('item features')
        item_features = cargar_datos_pickle(ruta_items)
        ruta_users = Entrada.elegir_archivo('user features')
        user_features = cargar_datos_pickle(ruta_users)


    def obtener_modelos(self):
        """
        Método obtener_modelos. Obtiene, entrena y guarda el modelo escogido.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, modelo, item_features, user_features
        
        # Se instancia un modelo con los parámetros por defecto
        modelo = LightFM(loss='warp')
        
        # Se entrena el modelo y se guarda
        if self.opcion_modelo == 1:
            modelo.fit(train, epochs=30, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo colaborativo')
        elif self.opcion_modelo == 2:
            modelo.fit(train, user_features=user_features, item_features=item_features, epochs=30, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo hibrido')
        else:
            modelo.fit(train, user_features=user_features, item_features=item_features, epochs=30, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo por contenido')


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
        no_components = lista_param[0]
        k = lista_param[1]
        n = lista_param[2]
        learning_schedule = lista_param[3]
        loss = lista_param[4]
        learning_rate = lista_param[5]
        rho = lista_param[6]
        epsilon = lista_param[7]
        item_alpha = lista_param[8]
        user_alpha = lista_param[9]
        max_sampled = lista_param[10]

        # Se instancia el modelo según los parámetros anteriores
        modelo = LightFM(no_components=no_components, k=k, n=n, learning_schedule=learning_schedule, loss=loss, learning_rate=learning_rate, rho=rho, 
            epsilon=epsilon, item_alpha=item_alpha, user_alpha=user_alpha, max_sampled=max_sampled)


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
        modelo = cargar_datos_pickle(ruta_modelo)


    def entrenar_modelo_gui(self):
        """
        Método entrenar_modelo_gui. Entrena el modelo escogido.

        Este método solo se utiliza en la interfaz web.
        """

        global train, modelo, item_features, user_features

        # Se entrena el modelo y se guarda
        if self.opcion_modelo == 1:
            modelo.fit(train, epochs=self.epochs, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo colaborativo')
        elif self.opcion_modelo == 2:
            modelo.fit(train, user_features=user_features, item_features=item_features, epochs=self.epochs, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo hibrido')
        else:
            modelo.fit(train, user_features=user_features, item_features=item_features, epochs=self.epochs, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo por contenido')
    

    def resultados_colaborativo(self):
        """
        Método resultados_colaboraivo. Obtiene los resultados del modelo colaborativo.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, test, modelo
        
        # Se obtienen los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()
        
        # Se imprimen los resultados
        imprimir_resultados_clasico(precision, auc, recall, reciprocal)
        

    def resultados_hibrido(self):
        """
        Método resultados_hibrido. Obtiene los resultados del modelo híbrido.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, test, modelo, user_features, item_features
        
        # Se obtienen los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, k=10, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, k=10, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        
        # Se imprimen los resultados
        imprimir_resultados_clasico(precision, auc, recall, reciprocal)
    

    def resultados_por_contenido(self):
        """
        Método resultados_por_contenido. Obtiene los resultados del modelo basado en contenido.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        global train, test, modelo, item_features, user_features
        
        # Se obtienen los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, k=10, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, k=10, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        
        # Se imprimen los resultados
        imprimir_resultados_clasico(precision, auc, recall, reciprocal)
    

    def obtener_resultados(self):
        """
        Método obtener_resultados. Obtiene los resultados del modelo escogido.

        Este método solo se utiliza en la interfaz de texto.
        """
        
        if self.opcion_modelo == 1:
            self.resultados_colaborativo()
        elif self.opcion_modelo == 2:
            self.resultados_hibrido()
        else:
            self.resultados_por_contenido()
    

    def obtener_metricas_gui(self):
        """
        Método obtener_metricas_gui. Obtiene las métricas del modelo escogido.

        Este método solo se utiliza en la interfaz web.
        """

        global train, test, modelo, item_features, user_features

        # Se guardan las métricas en un diccionario para su futura muestra en la interfaz web
        metricas = dict()

        # Se calculan las métricas
        if self.opcion_modelo == 1:
            precision = precision_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
            auc = auc_score(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()
            recall = recall_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
            reciprocal = reciprocal_rank(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()   
        else:
            precision = precision_at_k(modelo, test, train_interactions=train, k=10, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
            auc = auc_score(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
            recall = recall_at_k(modelo, test, train_interactions=train, k=10, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
            reciprocal = reciprocal_rank(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        
        # Se guardan las métricas en el diccionario y se formatea su salida
        metricas = {"Precisión k": format(precision, '.4f'), "AUC Score": format(auc, '.4f'), "Recall k": format(recall, '.4f'), "Ranking recíproco": format(reciprocal, '.4f')}
        
        return metricas


    def obtener_datos_conjunto_gui(self):
        """
        Método obtener_datos_conjunto_gui. Obtiene los datos del dataset escogido.

        Este método solo se utiliza en la interfaz web.
        """
        
        global train, test

        # Se guardan los datos en un diccionario para su futura muestra en la interfaz web
        datos = dict()

        # Se guarda el número de usuarios, de ítems y de valoraciones
        datos = {"Usuarios": train.shape[0], "Items": train.shape[1], "Valoraciones": train.getnnz()+test.getnnz()}
        
        return datos


    def obtener_id_maximo(self):
        """
        Método obtener_id_maximo. Obtiene el id del último usuario del dataset escogido.
        Se utiliza para indicarle al usuario de la aplicación que no debe pasarse de ese número.

        Este método solo se utiliza en la interfaz web.
        """

        global train

        return train.shape[0]
    
    
    def obtener_predicciones(self, usuario):
        """
        Método obtener_predicciones. Obtiene las 20 primeras predicciones para el usuario escogido.

        Este método solo se utiliza en la interfaz web.

        Parameters
        ----------

        usuario: int
            id del usuario cuyas predicciones se quieren obtener
        """

        global modelo, item_features, user_features

        if self.opcion_modelo == 1:
            scores = modelo.predict(usuario, np.arange(train.shape[1]), num_threads=self.CPU_THREADS)
        else:
            scores = modelo.predict(usuario, np.arange(train.shape[1]), item_features=item_features, user_features=user_features, num_threads=self.CPU_THREADS)
        
        # Se obtienen los ids de los ítems ordenando las predicciones de mayor a menor en función del score obtenido
        predicciones = np.argsort(-scores)
        
        # Se devuelven las 20 mejores predicciones
        return predicciones[:20]

