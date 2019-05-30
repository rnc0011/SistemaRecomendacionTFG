# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:34:06 2019

@author: Raúl Negro Carpintero
"""

def imprimir_resultados_clasico(precision=0.00, auc=0.00, recall=0.00, reciprocal=0.00):
    """
    Método imprimir_resultados_clasico. Imprimo las métricas del modelo.
    """
    
    print("Precision k: %.4f" % precision)
    print("AUC score: %.4f" % auc)
    print("Recall k: %.4f" % recall)
    print("Reciprocal rank: %.4f \n" % reciprocal)
    
def imprimir_resultados_dl(mrr=0.00, precision=0.00, recall=0.00, rmse=0.00):
    """
    Método imprimir_resultados_dl. Imprimo las métricas del modelo basado en aprendizaje profundo.
    """
    
    print("MRR: %.4f" % mrr)
    print("Precision k: %.4f" % precision)
    print("Recall k: %.4f" % recall)
    print("RMSE: %.4f" % rmse)
    
    