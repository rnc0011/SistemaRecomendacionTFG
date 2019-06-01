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
    file = open(ruta, 'rb').read()
    resultado = chardet.detect(file)
    encoding = resultado['encoding']
    return encoding

def obtener_sep(ruta):
    with open(ruta, newline='') as f:
        sniffer = csv.Sniffer()
        sniffer.preferred = [';', ',', '|']
        dialect = sniffer.sniff(f.read(2048))
    return dialect.delimiter

def elegir_archivo():
    root = Tk()
    tipos = [('Archivo CSV', '*.csv')]
    titulo = 'Selecciona un archivo'
    directorio = 'C:\\Downloads\\DatasetsTFG'
    root.filename = filedialog.askopenfilename(initialdir=directorio, title=titulo, filetypes=tipos)
    root.destroy()
    return root.filename

def leer_csv():
    sep = obtener_sep(ruta)
    encoding = obtener_encoding(ruta)
    if sep.isspace():
        dataframe = pd.read_csv(ruta, delim_whitespace=True, encoding=encoding)
    else:
        dataframe = pd.read_csv(ruta, sep=sep, encoding=encoding)
    return dataframe

def obtener_datos(self, opcion_modelo):
    global ratings_df
    global users_df
    global items_df

    if opcion_modelo == 2 or opcion_modelo == 3:
        print("Elige el archivo de usuarios")
        ruta_users = elegir_archivo()
        users_df = leer_csv(ruta_users)
        if opcion_modelo == 3:
            print("Elige el archivo de items")
            ruta_items = elegir_archivo()
            items_df = leer_csv(ruta_items)
    print("Elige el archivo de valoraciones")
    ruta_ratings = elegir_archivo()
    ratings_df = leer_csv(ruta_ratings)
    return items_df

    