# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:30:32 2019

@author: Raúl
"""

# Importo todo lo necesario
import os
import pandas as pd           

# Variables globales            
global opcion_dataset
global ratings_df
global users_df
global items_df

# Método leer_movielens. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_movielens():
    global ratings_df
    movielens_data_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Movielens', 'ml_data.csv')
    ratings_df = pd.read_csv(movielens_data_path, delim_whitespace=True, names=['Id Usuario', 'Id Película', 'Valoración', 'Fecha'])
    ratings_df.sort_values(['Id Usuario', 'Id Película'], inplace=True)

# Método leer_anime. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_anime():
    global ratings_df
    anime_data1_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Anime', 'ratings1.csv')
    anime_data2_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Anime', 'ratings2.csv')
    anime_data3_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Anime', 'ratings3.csv')
    anime_data4_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Anime', 'ratings4.csv')
    anime_data_1_df = pd.read_csv(anime_data1_path, sep=',', names=['Id Usuario', 'Id Anime', 'Valoración'], low_memory=False)
    anime_data_2_df = pd.read_csv(anime_data2_path, sep=',', names=['Id Usuario', 'Id Anime', 'Valoración'], low_memory=False)
    anime_data_3_df = pd.read_csv(anime_data3_path, sep=',', names=['Id Usuario', 'Id Anime', 'Valoración'], low_memory=False)
    anime_data_4_df = pd.read_csv(anime_data4_path, sep=',', names=['Id Usuario', 'Id Anime', 'Valoración'], low_memory=False)
    ratings_df = pd.concat([anime_data_1_df, anime_data_2_df, anime_data_3_df, anime_data_4_df])
    
# Método leer_book_crossing. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_book_crossing():
    global ratings_df
    bc_data_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Book-Crossing', 'BX-Book-Ratings.csv'),
    ratings_df = pd.read_csv(bc_data_path, sep=';', names=['Id Usuario','ISBN','Valoración'], encoding='cp1252', low_memory=False)
    ratings_df.sort_values(['Id Usuario'], inplace=True)

# Método leer_lastfm. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_lastfm():
    global ratings_df
    lf_data_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Last.FM', 'user_artists.csv')
    ratings_df = pd.read_csv(lf_data_path, sep='\\t', names=['Id Usuario','Id Artista','Veces escuchado'])
    
# Método leer_dating_agency. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_dating_agency():
    global ratings_df
    dating_data_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'DatingAgency', 'ratings.csv')
    ratings_df = pd.read_csv(dating_data_path, sep=',', names=['Id Usuario', 'Id Match', 'Valoración'], engine='python')

    
    