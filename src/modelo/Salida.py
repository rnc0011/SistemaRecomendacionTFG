# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:34:06 2019

@author: Raúl Negro Carpintero
"""

# Método imprimir_resultados_clasico. Imprimo las métricas del modelo.
def imprimir_resultados_clasico(precision, auc, recall, reciprocal):
    print('Precision k: %.4f' % precision)
    print('AUC score: %.4f' % auc)
    print('Recall k: %.4f' % recall)
    print('Reciprocal rank: %.4f \n' % reciprocal)
    
# Método imprimir_resultados_dl. Imprimo las métricas del modelo basado en aprendizaje profundo.
def imprimir_resultados_dl(rmse):
    print('RMSE: %.4f' %rmse)