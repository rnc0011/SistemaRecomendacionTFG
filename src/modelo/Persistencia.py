# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:40:44 2019

@author: Raúl Negro Carpintero
"""


# Se importa todo lo necesario
import torch
import pickle
import pandas as pd
from tkinter import *
from tkinter import filedialog  


def guardar_datos_pickle(datos, tipo):
    """
    Método guardar_datos_pickle. Guarda las matrices y los modelos producidos por LightFM además de las interacciones producidas por Spotlight.
    
    Parameters
    ----------
    
    datos: LightFM instance or np.float32 csr_matrix or Interactions instance
        modelo o matriz que se quiere guardar
    tipo: str
        cadena que especifica en la GUI qué archivo se quiere guardar
    """
    
    root = Tk()
    tipos = [('Archivo Pickle', '*.pickle')]
    titulo = 'Selecciona la carpeta donde guardar ' + tipo
    directorio = 'C:\\Downloads\\DatasetsTFG'
    extension = '.pickle'
    root.filename = filedialog.asksaveasfilename(initialdir=directorio, title=titulo, defaultextension=extension, filetypes=tipos)
    ruta = root.filename
    root.destroy()
    archivo_pickle = open(ruta, 'wb')
    pickle.dump(datos, archivo_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    archivo_pickle.close()


def cargar_datos_pickle(ruta_archivo):
    """
    Método cargar_datos_pickle. Carga las matrices y los modelos producidos por LightFM además de las interacciones producidas por Spotlight.
    
    Parameters
    ----------
    
    ruta_archivo: str
        ruta del modelo o matriz que se quiere cargar
    """

    archivo = open(ruta_archivo, 'rb')

    return pickle.load(archivo)


def guardar_modelos_dl(modelo, tipo):
    """
    Método guardar_modelos_dl. Guarda los modelos producidos por Spotlight.
    
    Parameters
    ----------
    
    modelo: ImplicitFactorizationModel, ExplicitFactorizationModelo or ImplicitSequenceModel instance
        modelo que se quiere guardar
    tipo: str
        cadena que especifica en la GUI qué archivo se quiere guardar
    """
    
    root = Tk()
    tipos = [('Archivo Pickle', '*.pickle')]
    titulo = 'Selecciona la carpeta donde guardar ' + tipo
    directorio = 'C:\\Downloads\\DatasetsTFG'
    extension = '.pickle'
    root.filename = filedialog.asksaveasfilename(initialdir=directorio, title=titulo, defaultextension=extension, filetypes=tipos)
    ruta = root.filename
    root.destroy()
    torch.save(modelo, ruta)


def cargar_modelo_dl(ruta_modelo):
    """
    Método cargar_modelo_dl. Carga los modelos producidos por Spotlight.
    
    Parameters
    ----------
    
    ruta_modelo: str
        ruta del modelo que se quiere cargar
    """

    return torch.load(ruta_modelo)    
    

def guardar_resultados(metricas):
    """
    Método guardar_resultados. Guarda las métricas producidas por los sistemas.
    
    Parameters
    ----------
    
    metricas: dict
        ruta del modelo que se quiere cargar
    """

    # Se obtiene el dataframe con las métricas
    dataframe = pd.DataFrame.from_dict(metricas)

    # Se obtiene la ruta donde se va a guardar el .csv
    root = Tk()
    tipos = [('Archivo CSV', '*.csv')]
    titulo = 'Selecciona la carpeta donde guardar el .csv con las métricas'
    directorio = 'C:\\Downloads\\DatasetsTFG'
    extension = '.csv'
    root.filename = filedialog.asksaveasfilename(initialdir=directorio, title=titulo, defaultextension=extension, filetypes=tipos)
    ruta = root.filename
    root.destroy()

    dataframe.to_csv(ruta, sep=';', header=True)

