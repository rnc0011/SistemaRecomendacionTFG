# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:02:09 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
from flask import Flask, render_template, request, redirect, url_for
from Forms import *

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


@app.route("/home/construir_modelo", methods=['GET', 'POST'])
def construir_modelo():
	form = DatasetForm(request.form)
	if request.method == 'GET':
		return render_template('construir_modelo.html', titulo='Construir Modelo', form=form)
	else:
		if form.menu.data == '1':
			return redirect(url_for('nuevo_dataset'))
		else:
			return redirect(url_for('datasets_prueba'))


@app.route("/home/construir_modelo/nuevo_dataset", methods=['GET','POST'])
def nuevo_dataset():
	form = NuevoDatasetForm(request.form)
	if request.method == 'GET':
		return render_template('nuevo_dataset.html', titulo='Nuevo Dataset', form=form)
	else:
		print(request.form['encoding'], request.form['separador'])
		# obtener los dataframes
		return redirect(url_for('elegir_modelo'))


@app.route("/home/construir_modelo/datasets_prueba", methods=['GET','POST'])
def datasets_prueba():
	form = DatasetsPruebaForm(request.form)
	if request.method == 'GET':
		return render_template('datasets_prueba.html', titulo='Datasets Prueba', form=form)
	else:
		# Lo que sea
		return redirect(url_for('elegir_modelo'))


@app.route("/home/construir_modelo/elegir_modelo", methods=['GET', 'POST'])
def elegir_modelo():
	form = ElegirModeloForm(request.form)
	if request.method == 'GET':
		return render_template('elegir_modelo.html', titulo='Elegir Modelo', form=form)
	else:
		if form.menu.data == '1':
			return redirect(url_for('elegir_modelo_clasico'))
		else:
			return redirect(url_for('elegir_modelo_dl'))


@app.route("/home/construir_modelo/elegir_modelo/elegir_modelo_clasico", methods=['GET', 'POST'])
def elegir_modelo_clasico():
	form = ElegirModeloClasicoForm(request.form)
	if request.method == 'GET':
		return render_template('elegir_modelo_clasico.html', titulo='Modelos Clásicos', form=form)
	else:
		return redirect(url_for('param_clasico'))


@app.route("/home/construir_modelo/elegir_modelo/elegir_modelo_dl", methods=['GET', 'POST'])
def elegir_modelo_dl():
	form = ElegirModeloDLForm(request.form)
	if request.method == 'GET':
		return render_template('elegir_modelo_dl.html', titulo='Modelos Deep Learning', form=form)
	else:
		return redirect(url_for('param_dl'))


@app.route("/home/construir_modelo/elegir_modelo/elegir_modelo_clasico/param_clasico", methods=['GET', 'POST'])
def param_clasico():
	form = ParamClasicoForm(request.form)
	return render_template('param_clasico.html', titulo='Parámetros Modelo Clásico', form=form)


@app.route("/home/construir_modelo/elegir_modelo/elegir_modelo_dl/param_dl", methods=['GET', 'POST'])
def param_dl():
	form = ParamDLForm(request.form)
	return render_template('param_dl.html', titulo='Parámetros Modelo Deep Learning', form=form)

if __name__ == '__main__':
	app.run(debug=True)


