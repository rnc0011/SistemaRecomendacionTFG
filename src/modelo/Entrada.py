# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:50:01 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import csv
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

def obtener_sep(ruta):
    """
    Método obtener_sep. Obtiene y devuelve el separador del arhcivo csv que se pasa por parámetro.
    
    Parameters
    ----------
    
    ruta: str
        ruta del archivo csv que se quiere utilizar.
        
    Returns
    -------
    
    sep: str
        separador del archivo csv que se para por parámetro.
    """
    
    with open(ruta, newline='') as f:
        sniffer = csv.Sniffer()
        sniffer.preferred = [';', ',', '|']
        dialect = sniffer.sniff(f.read(2048))
        sep = dialect.delimiter
    return sep

def elegir_archivo(tipo_archivo):
    """
    Método elegir_archivo. Muestra una interfaz con la que elegir los archivos csv.
    
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
    tipos = [('Archivo CSV', '*.csv')]
    titulo = 'Selecciona el archivo de ' + tipo_archivo
    directorio = 'C:\\Downloads\\DatasetsTFG'
    root.filename = filedialog.askopenfilename(initialdir=directorio, title=titulo, filetypes=tipos)
    ruta = root.filename
    root.destroy()
    return ruta

def leer_csv(ruta):
    """
    Método leer_csv. Lee el archivo csv cuya ruta se pasa por parámetro y obtiene su dataframe.
    
    Parameters
    ----------
    
    ruta: str
        ruta del archivo csv que se quiere utilizar.
        
    Returns
    -------
    
    dataframe: DataFrame
        dataframe del archivo csv.
    """
    
    sep = obtener_sep(ruta)
    encoding = obtener_encoding(ruta)
    if sep.isspace():
        dataframe = pd.read_csv(ruta, delim_whitespace=True, encoding=encoding)
    else:
        dataframe = pd.read_csv(ruta, sep=sep, encoding=encoding)
    return dataframe

def obtener_datos(opcion_modelo):
    """
    Método obtener_datos. Obtiene los dataframes requeridos por los modelos.
    
    Parameters
    ----------
    
    opcion_modelo: int
        modelo que se quiere obtener.
    """
    
    global ratings_df
    global users_df
    global items_df

    if opcion_modelo == 2 or opcion_modelo == 3:
        print("Elige el archivo de usuarios")
        ruta_users = elegir_archivo('usuarios')
        users_df = leer_csv(ruta_users)
        print("Elige el archivo de items")
        ruta_items = elegir_archivo('items')
        items_df = leer_csv(ruta_items)          
    print("Elige el archivo de valoraciones")
    ruta_ratings = elegir_archivo('valoraciones')
    ratings_df = leer_csv(ruta_ratings)
    ratings_df.sort_values([ratings_df.columns.values[0], ratings_df.columns.values[1]], inplace=True)

    