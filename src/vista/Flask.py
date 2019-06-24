# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:02:09 2019

@author: Raúl Negro Carpintero
"""

# Importo todo lo necesario
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from .Forms import *
from modelo import Entrada
from controlador import SistemaLightFM, SistemaSpotlight

UPLOAD_FOLDER = './uploads'

app = Flask('vista')
app.secret_key = 'development key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global sistema, modelo_clasico_dl, timestamps

@app.route("/home", methods=['GET','POST'])
def home():
	form = HomeForm(request.form)
	if request.method == 'GET':
		return render_template('home.html', titulo='Página Principal', form=form)
	else:
		if form.menu.data == '1':
			return redirect(url_for('elegir_modelo', path='/home', opcion_anterior=1))
		elif form.menu.data == '2':
			return redirect(url_for('elegir_modelo', path='/home', opcion_anterior=2))
		else:
			return redirect(url_for('anadir_valoraciones'))


@app.route("/<path:path>/<int:opcion_anterior>/elegir_modelo", methods=['GET','POST'])
def elegir_modelo(path, opcion_anterior):
	global modelo_clasico_dl
	
	form = ElegirModeloForm(request.form)
	if request.method == 'GET':
		return render_template('elegir_modelo.html', titulo='Elegir Modelo', form=form)
	else:
		if opcion_anterior == 1:
			if form.menu.data == '1':
				modelo_clasico_dl = 1
				return redirect(url_for('elegir_modelo_clasico', path=path+str(opcion_anterior)+'/elegir_modelo'))
			else:
				modelo_clasico_dl = 2
				return redirect(url_for('elegir_modelo_dl', path=path+str(opcion_anterior)+'/elegir_modelo'))
		else:
			if form.menu.data == '1':
				modelo_clasico_dl = 1
				return redirect(url_for('cargar_modelo_clasico', path=path+str(opcion_anterior)+'/elegir_modelo'))
			else:
				modelo_clasico_dl = 2
				return redirect(url_for('cargar_modelo_dl', path=path+str(opcion_anterior)+'/elegir_modelo'))


@app.route("/<path:path>/elegir_modelo_clasico", methods=['GET','POST'])
def elegir_modelo_clasico(path):
	form = ElegirModeloClasicoForm(request.form)
	if request.method == 'GET':
		return render_template('elegir_modelo_clasico.html', titulo='Modelos Clásicos', form=form)
	else:
		tipo_modelo = request.form['menu']
		return redirect(url_for('param_clasico', path=path+'/elegir_modelo_clasico', tipo=tipo_modelo))


@app.route("/<path:path>/elegir_modelo_dl", methods=['GET','POST'])
def elegir_modelo_dl():
	form = ElegirModeloDLForm(request.form)
	if request.method == 'GET':
		return render_template('elegir_modelo_dl.html', titulo='Modelos Deep Learning', form=form)
	else:
		tipo_modelo = int(request.form['menu'])
		return redirect(url_for('timestamps', path=path, tipo=tipo_modelo))


@app.route("/<path:path>/<int:tipo>/timestamps", methods=['GET','POST'])
def timestamps(path, tipo):
	global sistema, timestamps
	form = TimestampsForm(request.form)
	if request.method == 'GET':
		return render_template('timestamps.html', titulo='Usar timestamps', form=form)
	else:
		timestamps = int(request.form['menu'])
		sistema = SistemaSpotlight.SistemaSpotlight(tipo, timestamps)
		return redirect(url_for('param_dl', path=path+str(tipo)+'timestamps'))


@app.route("/<path:path>/<int:tipo>/param_clasico", methods=['GET','POST'])
def param_clasico(path, tipo):
	global sistema
	form = ParamClasicoForm(request.form)
	if request.method == 'GET':
		return render_template('param_clasico.html', titulo='Parámetros Modelo Clásico', form=form)
	else:
		epochs = int(request.form['epochs'])
		lista_param = list()
		lista_param.append(int(request.form['no_components']))
		lista_param.append(int(request.form['k']))
		lista_param.append(int(request.form['n']))
		lista_param.append(request.form['learning_schedule'])
		lista_param.append(request.form['loss'])
		lista_param.append(float(request.form['learning_rate']))
		lista_param.append(float(request.form['rho']))
		lista_param.append(float(request.form['epsilon']))
		lista_param.append(float(request.form['item_alpha']))
		lista_param.append(float(request.form['user_alpha']))
		lista_param.append(int(request.form['max_sampled']))
		sistema = SistemaLightFM.SistemaLightFM(tipo, epochs)
		sistema.obtener_modelo_gui(lista_param)
		return redirect(url_for('elegir_dataset', path=path+str(tipo)+'/param_clasico'))


@app.route("/<path:path>/param_dl", methods=['GET','POST'])
def param_dl(path):
	global sistema
	form = ParamDLForm(request.form)
	if request.method == 'GET':
		return render_template('param_dl.html', titulo='Parámetros Modelo Deep Learning', form=form)
	else:
		lista_param = list()
		lista_param.append(request.form['loss'])
		lista_param.append(int(request.form['embedding_dim']))
		lista_param.append(int(request.form['n_iter']))
		lista_param.append(int(request.form['batch_size']))
		lista_param.append(float(request.form['l2']))
		lista_param.append(float(request.form['learning_rate']))
		lista_param.append(request.form['representation'])
		sistema.obtener_modelo_gui(lista_param)
		return redirect(url_for('elegir_dataset', path=path+'/param_dl'))


@app.route("/<path:path>/elegir_dataset", methods=['GET','POST'])
def elegir_dataset(path):
	form = DatasetForm(request.form)
	if request.method == 'GET':
		return render_template('elegir_dataset.html', titulo='Elegir Dataset', form=form)
	else:
		if form.menu.data == '1':
			return redirect(url_for('nuevo_dataset', path=path+'/elegir_dataset'))
		else:
			return redirect(url_for('datasets_prueba', path=path+'/elegir_dataset'))


@app.route("/<path:path>/nuevo_dataset", methods=['GET','POST'])
def nuevo_dataset(path):
	global sistema, modelo_clasico_dl
	form = NuevoDatasetForm(request.form)
	if request.method == 'GET':
		return render_template('nuevo_dataset.html', titulo='Nuevo Dataset', form=form)
	else:
		# obtener el dataframe
		separador_ratings = request.form['separador_ratings']
		encoding_ratings = request.form['encoding_ratings']
		archivo_ratings = request.files['archivo_ratings']
		nombre_archivo_ratings = secure_filename(archivo_ratings.filename)
		archivo_ratings.save(os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo_ratings))
		dataframe_ratings = Entrada.leer_csv('./uploads/'+nombre_archivo_ratings, separador_ratings, encoding_ratings)
		separador_users = request.form['separador_users']
		encoding_users = request.form['encoding_users']
		archivo_users = request.files['archivo_users']
		nombre_archivo_users = secure_filename(archivo_users.filename)
		archivo_users.save(os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo_users))
		dataframe_users = Entrada.leer_csv('./uploads/'+nombre_archivo_users, separador_users, encoding_users)
		separador_items = request.form['separador_items']
		encoding_items = request.form['encoding_items']
		archivo_items = request.files['archivo_items']
		nombre_archivo_items = secure_filename(archivo_items.filename)
		archivo_items.save(os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo_items))
		dataframe_items = Entrada.leer_csv('./uploads/'+nombre_archivo_items, separador_items, encoding_items)
		if modelo_clasico_dl == 1:
			sistema.obtener_matrices_gui(dataframe_ratings, dataframe_users, dataframe_items)
			sistema.entrenar_modelo_gui()
		else:
			sistema.obtener_interacciones_gui(dataframe_ratings)
			sistema.entrenar_modelo_gui()
		return redirect(url_for('home'))


@app.route("/<path:path>/datasets_prueba", methods=['GET','POST'])
def datasets_prueba(path):
	global sistema, modelo_clasico_dl, timestamps
	
	form = DatasetsPruebaForm(request.form)
	if request.method == 'GET':
		return render_template('datasets_prueba.html', titulo='Datasets Prueba', form=form)
	else:
		if modelo_clasico_dl == 1:
			if form.menu.data == '1':
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_movielens.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_movielens.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_movielens.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_movielens.pickle')
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				sistema.entrenar_modelo_gui()
			elif form.menu.data == '2':
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_anime.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_anime.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_anime.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_anime.pickle')
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				sistema.entrenar_modelo_gui()
			elif form.menu.data == '3':
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_books.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_books.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_books.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_books.pickle')
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				sistema.entrenar_modelo_gui()
			elif form.menu.data == '4':
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_last.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_last.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_last.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_last.pickle')
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				sistema.entrenar_modelo_gui()
			elif form.menu.data == '5':
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_dating.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_dating.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_dating.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_dating.pickle')
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				sistema.entrenar_modelo_gui()
			else:
				sistema.cargar_otras_matrices_gui()	
				sistema.entrenar_modelo_gui()	
		else:
			if timestamps == 1:
				if form.menu.data == '1':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_movielens.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_movielens.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '2':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_anime.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_anime.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '3':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_books.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_books.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '4':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_last.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_last.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '5':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_dating.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_dating.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				else:
					sistema.cargar_otras_interacciones_gui()	
					sistema.entrenar_modelo_gui()	
			else:
				if form.menu.data == '1':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_movielens.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_movielens.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '2':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_anime.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_anime.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '3':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_books.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_books.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '4':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_last.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_last.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '5':
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_dating.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_dating.pickle')
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					sistema.entrenar_modelo_gui()
				else:
					sistema.cargar_otras_interacciones_gui()	
					sistema.entrenar_modelo_gui()
		return redirect(url_for('home'))


@app.route("/<path:path>/cargar_modelo_clasico", methods=['GET','POST'])
def cargar_modelo_clasico(path):
	global sistema
	
	form = CargarModeloClasicoForm(request.form)
	if request.method == 'GET':
		return render_template('cargar_modelo_clasico.html', titulo='Cargar Modelo Clásico', form=form)
	else:
		modelo = request.files['modelo']
		nombre_modelo = secure_filename(modelo.filename)
		ruta_modelo = os.path.join(app.config['UPLOAD_FOLDER'], modelo)
		train = request.files['archivo_train']
		nombre_train = secure_filename(train.filename)
		ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], nombre_train)
		test = request.files['archivo_test']
		nombre_test = secure_filename(test.filename)
		ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], nombre_test)
		items = request.files['archivo_items']
		nombre_items = secure_filename(items.filename)
		ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], nombre_items)
		users = request.files['archivo_users']
		nombre_users = secure_filename(users.filename)
		ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], nombre_users)
		sistema = SistemaLightFM.SistemaLightFM()
		sistema.cargar_modelo_gui(ruta_modelo)
		sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
		if form.menu.data == '1':
			return redirect(url_for('ver_metricas', path=path+'/cargar_modelo_clasico'))
		else:
			return redirect(url_for('elegir_usuario', path=path+'/cargar_modelo_clasico'))


@app.route("/<path:path>/cargar_modelo_dl", methods=['GET','POST'])
def cargar_modelo_dl(path):
	global sistema
	
	form = CargarModeloDLForm(request.form)
	if request.method == 'GET':
		return render_template('cargar_modelo_dl.html', titulo='Cargar Modelo DL', form=form)
	else:
		modelo = request.files['modelo']
		nombre_modelo = secure_filename(modelo.filename)
		ruta_modelo = os.path.join(app.config['UPLOAD_FOLDER'], modelo)
		train = request.files['archivo_train']
		nombre_train = secure_filename(train.filename)
		ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], nombre_train)
		test = request.files['archivo_test']
		nombre_test = secure_filename(test.filename)
		ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], nombre_test)
		sistema = SistemaSpotlight.SistemaSpotlight()
		sistema.cargar_modelo_gui(ruta_modelo)
		sistema.cargar_matrices_gui(ruta_train, ruta_test)
		if form.menu.data == '1':
			return redirect(url_for('ver_metricas', path=path+'/cargar_modelo_dl'))
		else:
			return redirect(url_for('elegir_usuario', path=path+'/cargar_modelo_dl'))


@app.route("/<path:path>/ver_metricas", methods=['GET','POST'])
def ver_metricas(path):
	form = MetricasForm(request.form)
	# hacer cosas
	if request.method == 'GET':
		return render_template('ver_metricas.html', titulo='Métricas', form=form)
	else:
		return redirect(url_for('home'))


@app.route("/<path:path>/elegir_usuario", methods=['GET','POST'])
def elegir_usuario(path):
	form = ElegirUsuarioForm(request.form)
	# hacer cosas
	if request.method == 'GET':
		return render_template('elegir_usuario.html', titulo='Elegir Usuario', form=form)
	else:
		return redirect(url_for('ver_predicciones', path=path+'/elegir_usuario'))


@app.route("/<path:path>/ver_predicciones", methods=['GET','POST'])
def ver_predicciones(path):
	form = PrediccionesForm(request.form)
	# hacer cosas
	if request.method == 'GET':
		return render_template('ver_predicciones.html', titulo='Predicciones', form=form)
	else:
		return redirect(url_for('home'))


@app.route("/home/añadir_valoraciones", methods=['GET','POST'])
def anadir_valoraciones():
	form = AnadirValoracionesForm(request.form)
	# hacer cosas
	if request.method == 'GET':
		return render_template('anadir_valoraciones.html', titulo='Añadir Valoraciones', form=form)
	else:
		return redirect(url_for('home'))


"""if __name__ == '__main__':
	app.run(debug=True)
"""

