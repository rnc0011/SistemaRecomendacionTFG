# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:50:01 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import os
import pandas as pd           

# Variables globales            
global ratings_df
global users_df
global items_df

# Método leer_movielens. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_movielens():
    global ratings_df
    global users_df
    global items_df
    movielens_data_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Movielens', 'ml_data.csv')
    movielens_users_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Movielens', 'ml_users.csv')
    movielens_items_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Movielens', 'ml_items.csv')
    ratings_df = pd.read_csv(movielens_data_path, delim_whitespace=True, names=['Id Usuario', 'Id Película', 'Valoración', 'Fecha'])
    ratings_df.sort_values(['Id Usuario', 'Id Película'], inplace=True)
    users_df = pd.read_csv(movielens_users_path, sep='|', names=['Id Usuario', 'Edad', 'Género', 'Ocupación', 'Código Postal'])
    items_df = pd.read_csv(movielens_items_path, sep='|', names=['Id Película', 'Título', 'Fecha de estreno', 'Fecha DVD', 'iMDB', 'Género desconocido', 'Acción', 
                                                                 'Aventura', 'Animación', 'Infantil', 'Comedia', 'Crimen', 'Docuemntal', 'Drama', 'Fantasía', 
                                                                 'Cine negro', 'Horror', 'Musical', 'Misterio', 'Romance', 'Ciencia ficción', 'Thriller', 'Bélico', 
                                                                 'Western'], encoding='latin-1')

# Método leer_anime. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_anime():
    global ratings_df
    global items_df
    anime_data1_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Anime', 'ratings1.csv')
    anime_data2_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Anime', 'ratings2.csv')
    anime_data3_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Anime', 'ratings3.csv')
    anime_data4_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Anime', 'ratings4.csv')
    anime_items_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Anime', 'anime.csv')
    items_df = pd.read_csv(anime_items_path, sep=',', names=['Id Anime', 'Título', 'Género', 'Tipo', 'Episodios', 'Valoración Media', 'Miembros'])
    items_df.sort_values(['Id Anime'], inplace=True)
    anime_data_1_df = pd.read_csv(anime_data1_path, sep=',', names=['Id Usuario', 'Id Anime', 'Valoración'], low_memory=False)
    anime_data_2_df = pd.read_csv(anime_data2_path, sep=',', names=['Id Usuario', 'Id Anime', 'Valoración'], low_memory=False)
    anime_data_3_df = pd.read_csv(anime_data3_path, sep=',', names=['Id Usuario', 'Id Anime', 'Valoración'], low_memory=False)
    anime_data_4_df = pd.read_csv(anime_data4_path, sep=',', names=['Id Usuario', 'Id Anime', 'Valoración'], low_memory=False)
    ratings_df = pd.concat([anime_data_1_df, anime_data_2_df, anime_data_3_df, anime_data_4_df])
    
# Método leer_book_crossing. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_book_crossing():
    global ratings_df
    global users_df
    global items_df
    bc_data_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Book-Crossing', 'BX-Book-Ratings.csv'),
    bc_users_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Book-Crossing', 'BX-Users.csv')
    bc_items_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Book-Crossing', 'BX-Books.csv')
    ratings_df = pd.read_csv(bc_data_path, sep=';', names=['Id Usuario','ISBN','Valoración'], encoding='cp1252', low_memory=False)
    ratings_df.sort_values(['Id Usuario'], inplace=True)
    users_df = pd.read_csv(bc_users_path, sep=';', names=['Id Usuario', 'Residencia', 'Edad'], encoding='cp1252')
    users_df = users_df.fillna(0)
    items_df = pd.read_csv(bc_items_path, sep=';', names=['ISBN','Título','Autor','Fecha de publicación','Editorial','URL S','URL M','URL L'], 
                           encoding='cp1252', low_memory=False)

# Método leer_lastfm. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_lastfm():
    global ratings_df
    global users_df
    global items_df
    lf_data_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Last.FM', 'user_artists.csv')
    lf_users_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Last.FM', 'user_friends.csv')
    lf_artists_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Last.FM', 'artists.csv')
    lf_generos_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'Last.FM', 'user_taggedartists-timestamps.csv')
    ratings_df = pd.read_csv(lf_data_path, sep='\\t', names=['Id Usuario','Id Artista','Veces escuchado'])
    users_df = pd.read_csv(lf_users_path, sep='\\t', names=['Id Usuario', 'Id Amigo'])
    lf_artists_df = pd.read_csv(lf_artists_path, sep='\\t', names=['Id Artista','Nombre','URL','URL Foto'])
    lf_artists_df = lf_artists_df.drop(['URL', 'URL Foto'], axis=1)
    lf_generos_df = pd.read_csv(lf_generos_path, sep='\\t', names=['Id Usuario','Id Artista','Id Genero','Timestamp'])
    lf_generos_df = lf_generos_df.drop(['Id Usuario', 'Timestamp'], axis=1)
    items_df = lf_artists_df.merge(lf_generos_df, left_on='Id Artista', right_on='Id Artista')
    
# Método leer_dating_agency. Lee el conjunto de datos y los almacena en varios dataframes.
def leer_dating_agency():
    global ratings_df
    global users_df
    dating_data_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'DatingAgency', 'ratings.csv')
    dating_users_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'DatasetsTFG', 'DatingAgency', 'gender.csv')
    ratings_df = pd.read_csv(dating_data_path, sep=',', names=['Id Usuario', 'Id Match', 'Valoración'], engine='python')
    users_df = pd.read_csv(dating_users_path, sep=',', names=['Id Usuario', 'Género'], engine='python')



    