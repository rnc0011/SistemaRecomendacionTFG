# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:34:06 2019

@author: Raúl Negro Carpintero
"""


def imprimir_resultados_clasico(precision, auc, recall, reciprocal):
    """
    Método imprimir_resultados_clasico. Imprime las métricas del modelo.

    Este método solo se utiliza en la interfaz de texto.
    
    Parameters
    ----------
    
    precision: np.float32
        precision at k del modelo.
    auc: np.float32
        auc score del modelo.
    recall: np.float64
        recall at k del modelo.
    reciprocal: np.float32
        ranking recíproco del modelo.
    """
    
    print("Precision k: %.4f" % precision)
    print("AUC score: %.4f" % auc)
    print("Recall k: %.4f" % recall)
    print("Reciprocal rank: %.4f \n" % reciprocal)
   

def imprimir_resultados_dl(mrr=0.00, precision=0.00, recall=0.00, rmse=0.00):
    """
    Método imprimir_resultados_dl. Imprimo las métricas del modelo basado en aprendizaje profundo.

    Este método solo se utiliza en la interfaz de texto.

    Parameters
    ----------
    
    mrr: np.float32
        mean reciprocal rank del modelo.
    precision: np.float32
        precision at k del modelo.
    recall: np.float64
        recall at k del modelo.
    rmse: float
        root mean square error del modelo.
    """
    
    print("MRR: %.4f" % mrr)
    print("Precision k: %.4f" % precision)
    print("Recall k: %.4f" % recall)
    print("RMSE: %.4f" % rmse)
    
    