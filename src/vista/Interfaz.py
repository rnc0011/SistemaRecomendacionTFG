# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:02:09 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import sys
sys.path.insert(0, 'C:\\Users\\Raúl\\Google Drive\\GitHub\\SistemaRecomendacionTFG\\src\\controlador')
#sys.path.insert(0, 'C:\\Users\\Raúl\\Google Drive\\GitHub\\SistemaRecomendacionTFG\\src\\modelo')
#import EntradaLightFM
import SistemaLightFM
import SistemaSpotlight

# Variables globales
#global opcion_dataset
#global opcion_modelo

# Método elegir_dataset. Muestra un menú para elegir el conjunto de datos a utilizar.
def elegir_dataset():
    #global opcion_dataset
    print("¿Qué conjunto de datos quieres utilizar?")
    while True:
        print("Introduce el número de la opción que elijas")
        print("MovieLens")
        print("Anime")
        print("Book Crossing")
        print("LastFM")
        print("Dating Agency")
        opcion_dataset = input()
        #print(opcion_dataset)
        if opcion_dataset in ['movielens', 'anime', 'books', 'lastfm', 'dating']:
            return opcion_dataset
            break
        else:
            print("No has introducido una opción válida")
            
# Método elegir_modelo_clasico. Muestra un menú para elegir el modelo clásico de recomendación a utilizar.
def elegir_modelo_clasico():
    #global opcion_modelo
    print("¿Qué modelo quieres utilizar?")
    while True:
        print("Introduce el número de la opción que elijas")
        print("1. Colaborativo")
        print("2. Híbrido")
        print("3. Por contenido")
        opcion_modelo = int(input())
        #print(opcion_modelo)
        if opcion_modelo > 0 and opcion_modelo < 4:
            return opcion_modelo
            break
        else:
            print("No has introducido una opción válida")
            
# Método elegir_modelo_dl. Muestra un menú para elegir el modelo deep learning de recomendación a utilizar.
def elegir_modelo_dl():
    #global opcion_modelo
    print("¿Qué modelo quieres utilizar?")
    while True:
        print("Introduce el número de la opción que elijas")
        print("1. Factorización explícito")
        print("2. Factorización implícito")
        print("3. Secuencia implícito")
        opcion_modelo = int(input())
        #print(opcion_modelo)
        if opcion_modelo > 0 and opcion_modelo < 4:
            return opcion_modelo
            break
        else:
            print("No has introducido una opción válida")
            
# Método main_clasico. Programa principal si la opción escogida es el modelo clásico.
def main_clasico():
    opcion_dataset = elegir_dataset()
    opcion_modelo = elegir_modelo_clasico()
    #EntradaLightFM.leer_csv(opcion_dataset)
    sistema = SistemaLightFM.SistemaLightFM(opcion_dataset, opcion_modelo)
    sistema.obtener_matrices()
    sistema.obtener_modelos()
    sistema.obtener_resultados()
    
# Método main_dl. Programa principal si la opción escogida es el modelo basado en aprendizaje profundo.
def main_dl():
    opcion_dataset = elegir_dataset()
    opcion_modelo = elegir_modelo_dl()
    sistema = SistemaSpotlight.SistemaSpotlight(opcion_dataset, opcion_modelo)
    sistema.obtener_interacciones()
    sistema.obtener_modelos()
    sistema.obtener_resultados()
    
# Método main. Método principal del programa.
def main():
    print("¿Qué modelo quieres utilizar?")
    while True:
        print("Introduce el número de la opción que elijas")
        print("1. Modelo clásico")
        print("2. Modelo aprendizaje profundo")
        opcion_inicial = int(input())
        #print(opcion_inicial)
        if opcion_inicial == 1:
            main_clasico()
            break
        elif opcion_inicial == 2:
            main_dl()
            break
        else:
            print("No has introducido una opción válida")

    
# Ejecución del programa.
main()


