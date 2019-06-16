# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:02:09 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
from flask import Flask, render_template, request, redirect, url_for
from Forms import HomeForm, DatasetForm, NuevoDatasetForm, DatasetsPruebaForm

app = Flask(__name__)
app.secret_key = 'development key'

@app.route("/home", methods=['GET','POST'])
def home():
	form = HomeForm(request.form)
	if form.menu.data == '1':
		return redirect(url_for('construir_modelo'))
	elif form.menu.data == '2':
		print("opcion 2")
	else:
		print("opcion 3")
	return render_template('home.html', titulo='Página Principal', form=form)


@app.route("/home/construir_modelo", methods=['GET','POST'])
def construir_modelo():
	form = DatasetForm(request.form)
	if form.menu.data == '1':
		return redirect(url_for('nuevo_dataset'))
	else:
		return redirect(url_for('datasets_prueba'))
	return render_template('construir_modelo.html', titulo='Construir Modelo', form=form)


@app.route("/home/construir_modelo/nuevo_dataset", methods=['GET','POST'])
def nuevo_dataset():
	form = NuevoDatasetForm(request.form)
	print(request.form['encoding'], request.form['separador'])
	return render_template('nuevo_dataset.html', methods=['GET','POST'], form=form)


@app.route("/home/construir_modelo/datasets_prueba", methods=['GET','POST'])
def datasets_prueba():
	form = DatasetsPruebaForm(request.form)
	return render_template('datasets_prueba.html', titulo='Cargar Dataset', form=form)


if __name__ == '__main__':
	app.run(debug=True)


