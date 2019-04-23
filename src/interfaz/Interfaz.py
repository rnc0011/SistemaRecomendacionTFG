# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:02:09 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import sys
sys.path.insert(0, 'C:\\Users\\Raúl\\Google Drive\\GitHub\\SistemaRecomendacionTFG\\src\\entrada')
sys.path.insert(0, 'C:\\Users\\Raúl\\Google Drive\\GitHub\\SistemaRecomendacionTFG\\src\\modelo')
import EntradaLightFM
import SistemaLightFM

# Variables globales
global opcion_dataset
global opcion_modelo

# Método elegir_dataset. Muestra un menú para elegir el conjunto de datos a utilizar.
def elegir_dataset():
    global opcion_dataset
    print("¿Qué conjunto de datos quieres utilizar?")
    while True:
        print("Introduce el número de la opción que elijas")
        print("1. MovieLens")
        print("2. Anime")
        print("3. Book Crossing")
        print("4. LastFM")
        print("5. Dating Agency")
        opcion_dataset = int(input())
        print(opcion_dataset)
        if opcion_dataset > 0 and opcion_dataset < 6:
            break
        else:
            print("No has introducido una opción válida")
            
# Método elegir_modelo. Muestra un menú para elegir el modelo de recomendación a utilizar.
def elegir_modelo():
    global opcion_modelo
    print("¿Qué modelo quieres utilizar?")
    while True:
        print("Introduce el número de la opción que elijas")
        print("1. Colaborativo")
        print("2. Híbrido")
        print("3. Por contenido")
        opcion_modelo = int(input())
        print(opcion_modelo)
        if opcion_modelo > 0 and opcion_modelo < 4:
            break
        else:
            print("No has introducido una opción válida")

# Método main. Método principal del programa.
def main():
    elegir_dataset()
    elegir_modelo()
    EntradaLightFM.leer_csv(opcion_dataset)
    sistema = SistemaLightFM.SistemaLightFM(opcion_dataset, opcion_modelo)
    sistema.obtener_matrices()
    sistema.obtener_modelo()
    sistema.obtener_resultados()
    
main()