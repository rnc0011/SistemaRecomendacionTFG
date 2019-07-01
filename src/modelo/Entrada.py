# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:50:01 2019

@author: Raúl Negro Carpintero
"""


# Se importa todo lo necesario
import chardet
import pandas as pd     
from tkinter import *
from tkinter import filedialog      


# Variables globales            
global ratings_df
global users_df
global items_df


def obtener_encoding(ruta):
    """
    Método obtener_encoding. Obtiene y devuelve el encoding del arhcivo csv que se pasa por parámetro.
    
    Parameters
    ----------
    
    ruta: str
        ruta del archivo csv que se quiere utilizar.
        
    Returns
    -------
    
    encoding: str
        encoding del archivo csv que se para por parámetro.
    """
    
    file = open(ruta, 'rb').read()
    resultado = chardet.detect(file)
    encoding = resultado['encoding']

    return encoding


def preguntar_encoding():
    """
    Método preguntar_encoding. Pregunta al usuario el encoding del arhcivo csv que quiere utilizar.

    Este método solo se utiliza en la interfaz de texto.
        
    Returns
    -------
    
    encoding: str
        encoding del archivo csv.
    """
    
    print("¿Qué encoding utiliza el archivo?")
    encoding = str(input())

    return encoding


def preguntar_sep():
    """
    Método preguntar_sep. Pregunta al usuario el separador del arhcivo csv que quiere utilizar.

    Este método solo se utiliza en la interfaz de texto.
    
    Returns
    -------
    
    sep: str
        separador del archivo csv.
    """
    
    print("¿Qué separador se utiliza?")
    sep = str(input())

    return sep


def elegir_archivo(tipo_archivo):
    """
    Método elegir_archivo. Muestra una interfaz con la que elegir los archivos csv.

    Este método solo se utiliza en la interfaz de texto.
    
    Parameters
    ----------
    
    tipo_archivo: str
        tipo del archivo csv que se quiere utilizar.
        
    Returns
    -------
    
    ruta: str
        ruta del archivo csv que se quiere utilizar.
    """
    
    root = Tk()
    tipos = [('Archivo CSV', '*.csv'), ('Archivo Pickle', '*.pickle')]
    titulo = 'Selecciona el archivo de ' + tipo_archivo
    directorio = 'C:\\Downloads\\DatasetsTFG'
    root.filename = filedialog.askopenfilename(initialdir=directorio, title=titulo, filetypes=tipos)
    ruta = root.filename
    root.destroy()

    return ruta

def leer_csv(ruta, sep, encoding):
    """
    Método leer_csv. Lee el archivo csv cuya ruta se pasa por parámetro y obtiene su dataframe.
    
    Parameters
    ----------
    
    ruta: str
        ruta del archivo csv que se quiere utilizar.
    sep: str
        separador del archivo csv que se quiere utilizar.
    encoding: str
        encoding del archivo csv que se quiere utilizar.
        
    Returns
    -------
    
    dataframe: DataFrame
        dataframe del archivo csv.
    """
    
    if sep.isspace():
        dataframe = pd.read_csv(ruta, delim_whitespace=True, encoding=encoding)
    else:
        dataframe = pd.read_csv(ruta, sep=sep, encoding=encoding)
    
    return dataframe


def obtener_datos():
    """
    Método obtener_datos. Obtiene los dataframes requeridos por los modelos.

    Este método solo se utiliza en la interfaz de texto.
    """
    
    global ratings_df
    global users_df
    global items_df

    print("Elige el archivo de usuarios")
    ruta_users = elegir_archivo('usuarios')
    sep = preguntar_sep()
    encoding = preguntar_encoding()
    users_df = leer_csv(ruta_users, sep, encoding)
    print("Elige el archivo de items")
    ruta_items = elegir_archivo('items')
    sep = preguntar_sep()
    encoding = preguntar_encoding()
    items_df = leer_csv(ruta_items, sep, encoding)          
    print("Elige el archivo de valoraciones")
    ruta_ratings = elegir_archivo('valoraciones')
    sep = preguntar_sep()
    encoding = preguntar_encoding()
    ratings_df = leer_csv(ruta_ratings, sep, encoding)
    ratings_df.sort_values([ratings_df.columns.values[0], ratings_df.columns.values[1]], inplace=True)

    