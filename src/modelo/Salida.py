# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:34:06 2019

@author: Raúl Negro Carpintero
"""

def imprimir_resultados(precision, auc, recall, reciprocal):
    print('Precision k: %.4f' % precision)
    print('AUC score: %.4f' % auc)
    print('Recall k: %.4f' % recall)
    print('Reciprocal rank: %.4f \n' % reciprocal)