# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:40:44 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import torch
import pickle
from tkinter import *
from tkinter import filedialog  

def guardar_datos_pickle(datos, tipo):
    """
    Método guardar_datos_pickle. Guarda tanto las matrices y los modelos producidos por LightFM además de las interacciones producidas por Spotlight.
    
    Parameters
    ----------
    
    datos: LightFM instance or np.float32 csr_matrix
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
    
    