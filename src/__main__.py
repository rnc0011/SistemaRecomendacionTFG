# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:31:59 2019

@author: Raúl
"""

print("¿Qué modelo quieres utilizar?")
print("1. Interfaz de texto")
print("2. Interfaz gráfica")
opcion = int(input())
if opcion == 1:
	from vista import Interfaz
	Interfaz.main()
elif opcion == 2:
	from vista import Flask
	Flask.app.run(debug=True)
else:
	print("No has introducido una opción válida")


