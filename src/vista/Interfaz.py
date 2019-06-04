# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:02:09 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import sys
sys.path.insert(0, 'C:\\Users\\Raúl\\Google Drive\\GitHub\\SistemaRecomendacionTFG\\src\\controlador')
import SistemaLightFM
import SistemaSpotlight
            
def elegir_modelo_clasico():
    """
    Método elegir_modelo_clasico. Muestra un menú para elegir el modelo clásico de recomendación a utilizar.
    """
    
    print("¿Qué modelo quieres utilizar?")
    while True:
        print("Introduce el número de la opción que elijas")
        print("1. Colaborativo")
        print("2. Híbrido")
        print("3. Por contenido")
        opcion_modelo_clasico = int(input())
        if opcion_modelo_clasico > 0 and opcion_modelo_clasico < 4:
            return opcion_modelo_clasico
            break
        else:
            print("No has introducido una opción válida")
            
def elegir_modelo_dl(opcion_time):
    """
    Método elegir_modelo_dl. Muestra un menú para elegir el modelo deep learning de recomendación a utilizar.
    
    Parameters
    ----------
    
    opcion_time: int
        opcion por si se quiere utilizar timestamps
    """
    
    print("¿Qué modelo quieres utilizar?")
    while True:
        print("Introduce el número de la opción que elijas")
        if opcion_time == 1:
            print("1. Factorización explícito")
            print("2. Factorización implícito")
            print("3. Secuencia implícito")
            opcion_modelo_dl = int(input())
            if opcion_modelo_dl > 0 and opcion_modelo_dl < 4:
                return opcion_modelo_dl
                break
            else:
                print("No has introducido una opción válida")
        else:
            print("1. Factorización explícito")
            print("2. Factorización implícito")
            opcion_modelo_dl = int(input())
            if opcion_modelo_dl > 0 and opcion_modelo_dl < 3:
                return opcion_modelo_dl
                break
            else:
                print("No has introducido una opción válida")
            
def main_clasico():
    """
    Método main_clasico. Programa principal si la opción escogida es el modelo clásico.
    """
    
    opcion_modelo_clasico = elegir_modelo_clasico()
    sistema = SistemaLightFM.SistemaLightFM(opcion_modelo_clasico)
    sistema.obtener_matrices()
    sistema.obtener_modelos()
    sistema.obtener_resultados()
    
def main_dl():
    """
    Método main_dl. Programa principal si la opción escogida es el modelo basado en aprendizaje profundo.
    """
    
    print("¿El conjunto de datos tiene timestamps?")
    while True:
        print("1. Sí")
        print("2. No")
        opcion_time = int(input())
        if opcion_time == 1 or opcion_time == 2:
            break
        else:
            print("No has introducido una opción válida")
    opcion_modelo_dl = elegir_modelo_dl(opcion_time)
    sistema = SistemaSpotlight.SistemaSpotlight(opcion_modelo_dl, opcion_time)
    sistema.obtener_interacciones()
    sistema.obtener_modelos()
    sistema.obtener_resultados()
    
def main():
    """
    Método main. Método principal del programa.
    """
    
    print("¿Qué modelo quieres utilizar?")
    while True:
        print("Introduce el número de la opción que elijas")
        print("1. Modelo clásico")
        print("2. Modelo aprendizaje profundo")
        opcion_inicial = int(input())
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




