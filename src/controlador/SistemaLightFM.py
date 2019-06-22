# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:45:27 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import multiprocessing
from modelo import Entrada
from modelo.Salida import imprimir_resultados_clasico
from modelo.Persistencia import guardar_datos_pickle
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
        Método obtener_matrices. Obtiene las matrices necesarias para la creación de los modelos con LightFM.
        """
        
        global train, test, modelo, item_features, user_features
        
        # Obtengo los datos
        Entrada.obtener_datos()
        ratings_df = Entrada.ratings_df

        # Obtengo las matrices
        dataset = Dataset()
        if self.opcion_modelo == 1:
            dataset.fit(ratings_df[ratings_df.columns.values[0]], ratings_df[ratings_df.columns.values[1]])
            (interacciones, pesos) = dataset.build_interactions((row[ratings_df.columns.values[0]],
                                                                 row[ratings_df.columns.values[1]],
                                                                 row[ratings_df.columns.values[2]]) 
                                                                for index,row in ratings_df.iterrows())
        else:
            users_df = Entrada.users_df
            items_df = Entrada.items_df
            dataset.fit(users_df[users_df.columns.values[0]], items_df[items_df.columns.values[0]],
                       user_features=users_df[users_df.columns.values[1]], item_features=items_df[items_df.columns.values[1]])
            (interacciones, pesos) = dataset.build_interactions((row[ratings_df.columns.values[0]],
                                                                 row[ratings_df.columns.values[1]],
                                                                 row[ratings_df.columns.values[2]]) 
                                                                for index,row in ratings_df.iterrows())
            item_features = dataset.build_item_features((row[items_df.columns.values[0]], [row[items_df.columns.values[1]]]) for index, row in items_df.iterrows())
            user_features = dataset.build_user_features((row[users_df.columns.values[0]], [row[users_df.columns.values[1]]]) for index, row in users_df.iterrows())
            # Guardo los datos
            print("Guarda la matriz de item features")
            guardar_datos_pickle(item_features, 'la matriz de item features')
            print("Guarda la matriz de user features")
            guardar_datos_pickle(user_features, 'la matriz de user feautures')
            
        train, test = random_train_test_split(interacciones, test_percentage=0.2)
        
        # Guardo los datos
        print("Guarda la matriz de entrenamiento")
        guardar_datos_pickle(train, 'la matriz de entrenamiento')
        print("Guarda la matriz de test")
        guardar_datos_pickle(test, 'la matriz de test')
            
    def obtener_matrices_gui(self, ratings_df, users_df, items_df):
        global train, test, item_features, user_features

        # Obtengo las matrices
        dataset = Dataset()
        if self.opcion_modelo == 1:
            dataset.fit(ratings_df[ratings_df.columns.values[0]], ratings_df[ratings_df.columns.values[1]])
            (interacciones, pesos) = dataset.build_interactions((row[ratings_df.columns.values[0]],
                                                                 row[ratings_df.columns.values[1]],
                                                                 row[ratings_df.columns.values[2]]) 
                                                                for index,row in ratings_df.iterrows())
        else:
            dataset.fit(users_df[users_df.columns.values[0]], items_df[items_df.columns.values[0]],
                       user_features=users_df[users_df.columns.values[1]], item_features=items_df[items_df.columns.values[1]])
            (interacciones, pesos) = dataset.build_interactions((row[ratings_df.columns.values[0]],
                                                                 row[ratings_df.columns.values[1]],
                                                                 row[ratings_df.columns.values[2]]) 
                                                                for index,row in ratings_df.iterrows())
            item_features = dataset.build_item_features((row[items_df.columns.values[0]], [row[items_df.columns.values[1]]]) for index, row in items_df.iterrows())
            user_features = dataset.build_user_features((row[users_df.columns.values[0]], [row[users_df.columns.values[1]]]) for index, row in users_df.iterrows())
            # Guardo los datos
            print("Guarda la matriz de item features")
            guardar_datos_pickle(item_features, 'la matriz de item features')
            print("Guarda la matriz de user features")
            guardar_datos_pickle(user_features, 'la matriz de user feautures')
            
        train, test = random_train_test_split(interacciones, test_percentage=0.2)
        
        # Guardo los datos
        print("Guarda la matriz de entrenamiento")
        guardar_datos_pickle(train, 'la matriz de entrenamiento')
        print("Guarda la matriz de test")
        guardar_datos_pickle(test, 'la matriz de test')

    def obtener_modelos(self):
        """
        Método obtener_modelos. Obtiene el modelo escogido.
        """
        
        global train, modelo, item_features, user_features
        
        # Obtengo el modelo
        modelo = LightFM(loss='warp')
        
        # Entreno y guardo el modelo
        if self.opcion_modelo == 1:
            modelo.fit(train, epochs=30, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo colaborativo')
        elif self.opcion_modelo == 2:
            modelo.fit(train, item_features=item_features, epochs=30, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo hibrido')
        else:
            modelo.fit(train, user_features=user_features, item_features=item_features, epochs=30, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo por contenido')

    def obtener_modelo_gui(self, lista_param):
        global modelo

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
        modelo = LightFM(no_components=no_components, k=k, n=n, learning_schedule=learning_schedule, loss=loss, learning_rate=learning_rate, rho=rho, 
            epsilon=epsilon, item_alpha=item_alpha, user_alpha=user_alpha, max_sampled=max_sampled)

    def entrenar_modelo_gui(self):
        global modelo

        # Entreno y guardo el modelo
        if self.opcion_modelo == 1:
            modelo.fit(train, epochs=self.epochs, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo colaborativo')
        elif self.opcion_modelo == 2:
            modelo.fit(train, item_features=item_features, epochs=self.epochs, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo hibrido')
        else:
            modelo.fit(train, user_features=user_features, item_features=item_features, epochs=self.epochs, num_threads=self.CPU_THREADS)
            guardar_datos_pickle(modelo, 'el modelo por contenido')
    
    def resultados_colaborativo(self):
        """
        Método resultados_colaboraivo. Obtiene los resultados del modelo colaborativo.
        """
        
        global train, test, modelo
        
        # Obtención de los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()
        
        imprimir_resultados_clasico(precision, auc, recall, reciprocal)
        
    def resultados_hibrido(self):
        """
        Método resultados_hibrido. Obtiene los resultados del modelo híbrido.
        """
        
        global train, test, modelo, item_features
        
        # Obtención de los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        
        imprimir_resultados_clasico(precision, auc, recall, reciprocal)
    
    def resultados_por_contenido(self):
        """
        Método resultados_por_contenido. Obtiene los resultados del modelo por contenido.
        """
        
        global train, test, modelo, item_features, user_features
        
        # Obtención de los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        
        imprimir_resultados_clasico(precision, auc, recall, reciprocal)
    
    def obtener_resultados(self):
        """
        Método obtener_resultados. Obtiene los resultados en función del modelo escogido.
        """
        
        if self.opcion_modelo == 1:
            self.resultados_colaborativo()
        elif self.opcion_modelo == 2:
            self.resultados_hibrido()
        else:
            self.resultados_por_contenido()
    
    
    
    
    