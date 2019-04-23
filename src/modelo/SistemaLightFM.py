# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:45:27 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import multiprocessing
#import pickle
#import os
import EntradaLightFM
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import reciprocal_rank
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split

# Clase SistemaLightFM.
class SistemaLightFM:
    
    # Variables globales
    global train
    global test
    global modelo
    global item_features
    global user_features
    
    # Constantes
    CPU_THREADS = multiprocessing.cpu_count()
    
    # Método __init__. Inicializa la clase con el conjunto de datos y el modelo escogidos.
    def __init__(self, opcion_dataset, opcion_modelo):
        self.opcion_dataset = opcion_dataset
        self.opcion_modelo = opcion_modelo
    
    # Método matrices_movilens. Crea las matrices de movielens con las que poder utilizar los modelos.    
    def matrices_movielens(self):
        global train
        global test
        global item_features
        global user_features
        
        # Obtención de los dataset
        ml_dataset = Dataset()
        ml_dataset.fit(EntradaLightFM.ratings_df['Id Usuario'], EntradaLightFM.ratings_df['Id Película'])
        ml_dataset.fit_partial(users=EntradaLightFM.users_df['Id Usuario'], items=EntradaLightFM.items_df['Id Película'], 
                               user_features=EntradaLightFM.users_df['Género'], item_features=EntradaLightFM.items_df['Título'])
        
        # Obtención de las matrices
        (ml_interactions, ml_weights) = ml_dataset.build_interactions((row['Id Usuario'], row['Id Película'], row['Valoración']) for index, row in EntradaLightFM.ratings_df.iterrows())
        ml_item_features = ml_dataset.build_item_features((row['Id Película'], [row['Título']]) for index, row in EntradaLightFM.items_df.iterrows())
        ml_user_features = ml_dataset.build_user_features((row['Id Usuario'], [row['Género']]) for index, row in EntradaLightFM.users_df.iterrows())
        
        # División de los datos
        ml_train, ml_test = random_train_test_split(ml_interactions, test_percentage=0.2)
        train, test, item_features, user_features = ml_train, ml_test, ml_item_features, ml_user_features
        
    # Método matrices_anime. Crea las matrices de anime con las que poder utilizar los modelos. 
    def matrices_anime(self):
        global train
        global test
        global item_features
        
        # Obtención de los dataset
        anime_dataset = Dataset()
        anime_dataset.fit(EntradaLightFM.ratings_df['Id Usuario'], EntradaLightFM.ratings_df['Id Anime'])
        anime_dataset.fit_partial(items=EntradaLightFM.items_df['Id Anime'], item_features=EntradaLightFM.items_df['Título'])
        
        # Obtención de las matrices
        (anime_interactions, anime_weights) = anime_dataset.build_interactions((row['Id Usuario'], row['Id Anime'], row['Valoración']) for index, row in EntradaLightFM.ratings_df.iterrows())
        anime_item_features = anime_dataset.build_item_features((row['Id Anime'], [row['Título']]) for index, row in EntradaLightFM.items_df.iterrows())
        
        # División de los datos
        anime_train, anime_test = random_train_test_split(anime_interactions, test_percentage=0.2)
        train, test, item_features = anime_train, anime_test, anime_item_features
        
    # Método matrices_book_crossing. Crea las matrices de book crossing con las que poder utilizar los modelos. 
    def matrices_book_crossing(self):
        global train
        global test
        global item_features
        global user_features
        
        # Obtención de los dataset
        bc_dataset = Dataset()
        bc_dataset.fit(EntradaLightFM.ratings_df['Id Usuario'], EntradaLightFM.ratings_df['ISBN'])
        bc_dataset.fit_partial(users=EntradaLightFM.users_df['Id Usuario'], items=EntradaLightFM.items_df['ISBN'], 
                               user_features=EntradaLightFM.users_df['Edad'], item_features=EntradaLightFM.items_df['Título'])
        
        # Obtención de las matrices
        (bc_interactions, bc_weights) = bc_dataset.build_interactions((row['Id Usuario'], row['ISBN'], row['Valoración']) for index, row in EntradaLightFM.ratings_df.iterrows())
        bc_item_features = bc_dataset.build_item_features((row['ISBN'], [row['Título']]) for index, row in EntradaLightFM.items_df.iterrows())
        bc_user_features = bc_dataset.build_user_features((row['Id Usuario'], [row['Edad']]) for index, row in EntradaLightFM.users_df.iterrows())
        
        # División de los datos
        bc_train, bc_test = random_train_test_split(bc_interactions, test_percentage=0.2)
        train, test, item_features, user_features = bc_train, bc_test, bc_item_features, bc_user_features
        
    # Método matrices_lastfm. Crea las matrices de lastfm con las que poder utilizar los modelos. 
    def matrices_lastfm(self):
        global train
        global test
        global item_features
        global user_features
        
        # Obtención de los dataset
        lf_dataset = Dataset()
        lf_dataset.fit(EntradaLightFM.ratings_df['Id Usuario'], EntradaLightFM.ratings_df['Id Artista'])
        lf_dataset.fit_partial(users=EntradaLightFM.users_df['Id Usuario'], items=EntradaLightFM.items_df['Id Artista'], 
                               user_features=EntradaLightFM.users_df['Id Amigo'], item_features=EntradaLightFM.items_df['Nombre'])
        
        # Obtención de las matrices
        (lf_interactions, lf_weights) = lf_dataset.build_interactions((row['Id Usuario'], row['Id Artista'], row['Veces escuchado']) for index, row in EntradaLightFM.ratings_df.iterrows())
        lf_item_features = lf_dataset.build_item_features((row['Id Artista'], [row['Nombre']]) for index, row in EntradaLightFM.items_df.iterrows())
        lf_user_features = lf_dataset.build_user_features((row['Id Usuario'], [row['Id Amigo']]) for index, row in EntradaLightFM.users_df.iterrows())
        
        # División de los datos
        lf_train, lf_test = random_train_test_split(lf_interactions, test_percentage=0.2)
        train, test, item_features, user_features = lf_train, lf_test, lf_item_features, lf_user_features
        
    # Método matrices_dating_agency. Crea las matrices de dating agency con las que poder utilizar los modelos. 
    def matrices_dating_agency(self):
        global train
        global test
        global item_features
        global user_features
        
        # Obtención de los dataset
        dating_dataset = Dataset()
        dating_dataset.fit(EntradaLightFM.ratings_df['Id Usuario'], EntradaLightFM.ratings_df['Id Match'])
        dating_dataset.fit_partial(users=EntradaLightFM.users_df['Id Usuario'], items=EntradaLightFM.users_df['Id Usuario'], user_features=EntradaLightFM.users_df['Género'], item_features=EntradaLightFM.users_df['Género'])
        
        # Obtención de las matrices
        (dating_interactions, dating_weights) = dating_dataset.build_interactions((row['Id Usuario'], row['Id Match'], row['Valoración']) for index, row in EntradaLightFM.ratings_df.iterrows())
        dating_item_features = dating_dataset.build_item_features((row['Id Usuario'], [row['Género']]) for index, row in EntradaLightFM.users_df.iterrows())
        dating_user_features = dating_dataset.build_user_features((row['Id Usuario'], [row['Género']]) for index, row in EntradaLightFM.users_df.iterrows())
        
        # División de los datos
        dating_train, dating_test = random_train_test_split(dating_interactions, test_percentage=0.2)
        train, test, item_features, user_features = dating_train, dating_test, dating_item_features, dating_user_features
        
    # Método obtener:matrices. Crea las matrices con las que poder utilizar los modelos en función de la opción escogida.
    def obtener_matrices(self):
        if self.opcion_dataset == 1:
            self.matrices_movielens()
        elif self.opcion_dataset == 2:
            self.matrices_anime()
        elif self.opcion_dataset == 3:
            self.matrices_book_crossing()
        elif self.opcion_dataset == 4:
            self.matrices_lastfm()
        else:
            self.matrices_dating_agency()
    
    # Método modelo_colaborativo. Crea el modelo colaborativo.
    def modelo_colaborativo(self):
        global train
        global modelo
        
        # Obtención del modelo
        modelo = LightFM(loss='warp')
        modelo.fit(train, epochs=30, num_threads=self.CPU_THREADS)
        
    # Método modelo_hibrido. Crea el modelo híbrido.
    def modelo_hibrido(self):
        global train
        global modelo
        global item_features
        
        # Obtención del modelo
        modelo = LightFM(loss='warp')
        modelo.fit(train, item_features=item_features, epochs=30, num_threads=self.CPU_THREADS)
        
    # Método modelo_por_contenido. Crea el modelo por contenido.
    def modelo_por_contenido(self):
        global train
        global modelo
        global item_features
        global user_features
        
        # Obtención del modelo
        modelo = LightFM(loss='warp')
        modelo.fit(train, user_features=user_features, item_features=item_features, epochs=30, num_threads=self.CPU_THREADS)
    
    # Método obtener_modelo. Crea los modelos en función de la opción escogida.
    def obtener_modelo(self):
        if self.opcion_modelo == 1:
            self.modelo_colaborativo()
        elif self.opcion_modelo == 2:
            self.modelo_hibrido()
        else:
            self.modelo_por_contenido()
            
    # Método resultados_colaboraivo. Obtiene los resultados del modelo colaborativo.
    def resultados_colaborativo(self):
        global train
        global test
        global modelo
        
        # Obtención de los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, num_threads=self.CPU_THREADS).mean()
        
        print(precision, auc, recall, reciprocal)
        
    # Método resultados_hibrido. Obtiene los resultados del modelo híbrido.
    def resultados_hibrido(self):
        global train
        global test
        global modelo
        global item_features
        
        # Obtención de los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        
        print(precision, auc, recall, reciprocal)
    
    # Método resultados_por_contenido. Obtiene los resultados del modelo por contenido.
    def resultados_por_contenido(self):
        global train
        global test
        global modelo
        global item_features
        global user_features
        
        # Obtención de los resultados
        precision = precision_at_k(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        auc = auc_score(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        recall = recall_at_k(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, k=10, num_threads=self.CPU_THREADS).mean()
        reciprocal = reciprocal_rank(modelo, test, train_interactions=train, user_features=user_features, item_features=item_features, num_threads=self.CPU_THREADS).mean()
        
        print(precision, auc, recall, reciprocal)
    
    # Método obtener_resultados. Obtiene los resultados en función del modelo escogido.
    def obtener_resultados(self):
        if self.opcion_modelo == 1:
            self.resultados_colaborativo()
        elif self.opcion_modelo == 2:
            self.resultados_hibrido()
        else:
            self.resultados_por_contenido()
    
    
    
    
    