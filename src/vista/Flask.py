# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:02:09 2019

@author: Raúl Negro Carpintero
"""


# Se importa todo lo necesario
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from .Forms import *
from modelo import Entrada
from controlador import SistemaLightFM, SistemaSpotlight

# Constantes
UPLOAD_FOLDER = './uploads'


app = Flask('vista')
app.secret_key = 'development key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Variables globales
global sistema, modelo_clasico_dl, timestamps


@app.route("/home", methods=['GET','POST'])
def home():
	"""
    Método home. Muestra la página principal de la interfaz web.

    Este método solo se utiliza en la interfaz web.
    """
     
    # Se genera el formulario
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
	"""
    Método elegir_modelo. Muestra la página de selección del tipo de modelo.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    opcion_anterior: int
    	opcion del tipo de modelo seleccionado
    """
   
	global modelo_clasico_dl
	
	# Se genera el formulario
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
	"""
    Método elegir_modelo_clasico. Muestra la página de selección del tipo de modelo clásico.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    """
  	
  	# Se genera el formulario
	form = ElegirModeloClasicoForm(request.form)

	if request.method == 'GET':
		return render_template('elegir_modelo_clasico.html', titulo='Modelos Clásicos', form=form)
	else:
		tipo_modelo = request.form['menu']
		return redirect(url_for('param_clasico', path=path+'/elegir_modelo_clasico', tipo=tipo_modelo))


@app.route("/<path:path>/elegir_modelo_dl", methods=['GET','POST'])
def elegir_modelo_dl(path):
	"""
    Método elegir_modelo_dl. Muestra la página de selección del tipo de modelo basado en aprendizaje profundo.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    """
  	
  	# Se genera el formulario
	form = ElegirModeloDLForm(request.form)

	if request.method == 'GET':
		return render_template('elegir_modelo_dl.html', titulo='Modelos Deep Learning', form=form)
	else:
		tipo_modelo = int(request.form['menu'])
		return redirect(url_for('timestamps', path=path, tipo=tipo_modelo))


@app.route("/<path:path>/<int:tipo>/timestamps", methods=['GET','POST'])
def timestamps(path, tipo):
	"""
    Método timestamps. Muestra la página donde indicar si el dataset que se va a utilizar lleva timestamps o no.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    tipo: int
		tipo de modelo de Spotlight
    """
  	
	global sistema, timestamps

	# Se genera el formulario
	form = TimestampsForm(request.form)

	if request.method == 'GET':
		return render_template('timestamps.html', titulo='Usar timestamps', form=form)
	else:
		timestamps = int(request.form['menu'])
		# Se instancia el sistema de Spotligth
		sistema = SistemaSpotlight.SistemaSpotlight(tipo, timestamps)
		return redirect(url_for('param_dl', path=path+str(tipo)+'timestamps'))


@app.route("/<path:path>/<int:tipo>/param_clasico", methods=['GET','POST'])
def param_clasico(path, tipo):
	"""
    Método param_clasico. Muestra la página donde indicar los parámetros con los que obtener el modelo de LightFM.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    tipo: int
		tipo de modelo de LightFM
    """
  	
	global sistema

	# Se genera el formulario
	form = ParamClasicoForm(request.form)

	if request.method == 'GET':
		return render_template('param_clasico.html', titulo='Parámetros Modelo Clásico', form=form)
	else:
		# Se recogen los parámetros
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
		# Se instancia el sistema de LightFM
		sistema = SistemaLightFM.SistemaLightFM(tipo, epochs)
		# Se obtiene el modelo de LightFM
		sistema.obtener_modelo_gui(lista_param)
		return redirect(url_for('elegir_dataset', path=path+str(tipo)+'/param_clasico'))


@app.route("/<path:path>/param_dl", methods=['GET','POST'])
def param_dl(path):
	"""
    Método param_dl. Muestra la página donde indicar los parámetros con los que obtener el modelo de Spotlight.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    """
  	
	global sistema

	# Se genera el formulario
	form = ParamDLForm(request.form)

	if request.method == 'GET':
		return render_template('param_dl.html', titulo='Parámetros Modelo Deep Learning', form=form)
	else:
		# Se recogen los parámetros
		lista_param = list()
		lista_param.append(request.form['loss'])
		lista_param.append(int(request.form['embedding_dim']))
		lista_param.append(int(request.form['n_iter']))
		lista_param.append(int(request.form['batch_size']))
		lista_param.append(float(request.form['l2']))
		lista_param.append(float(request.form['learning_rate']))
		lista_param.append(request.form['representation'])
		# Se obtiene el modelo de Spotlight
		sistema.obtener_modelo_gui(lista_param)
		return redirect(url_for('elegir_dataset', path=path+'/param_dl'))


@app.route("/<path:path>/elegir_dataset", methods=['GET','POST'])
def elegir_dataset(path):
	"""
    Método elegir_dataset. Muestra la página donde escoger entre usar un dataset ya guardado o uno nuevo.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    """
  	
	# Se genera el formulario
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
	"""
    Método nuevo_dataset. Muestra la página donde seleccionar los archivos que se quieren utilizar.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    """
  	
	global sistema, modelo_clasico_dl

	# Se genera el formulario
	form = NuevoDatasetForm(request.form)

	if request.method == 'GET':
		return render_template('nuevo_dataset.html', titulo='Nuevo Dataset', form=form)
	else:
		# Se recogen los separadores, los encodings y las rutas de los archivos que se quieren utilizar
		separador_ratings = request.form['separador_ratings']
		encoding_ratings = request.form['encoding_ratings']
		archivo_ratings = request.files['archivo_ratings']
		nombre_archivo_ratings = secure_filename(archivo_ratings.filename)
		archivo_ratings.save(os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo_ratings))
		
		separador_users = request.form['separador_users']
		encoding_users = request.form['encoding_users']
		archivo_users = request.files['archivo_users']
		nombre_archivo_users = secure_filename(archivo_users.filename)
		archivo_users.save(os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo_users))

		separador_items = request.form['separador_items']
		encoding_items = request.form['encoding_items']
		archivo_items = request.files['archivo_items']
		nombre_archivo_items = secure_filename(archivo_items.filename)
		archivo_items.save(os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo_items))

		if modelo_clasico_dl == 1:
			# Se obtienen las matrices de LightFM
			sistema.obtener_matrices_gui('./uploads/'+nombre_archivo_ratings, separador_ratings, encoding_ratings, 
				'./uploads/'+nombre_archivo_users, separador_users, encoding_users, 
				'./uploads/'+nombre_archivo_items, separador_items, encoding_items)
			# Se entrena el modelo de LightFM
			sistema.entrenar_modelo_gui()
		else:
			# Se obtienen las interacciones de Spotlight
			sistema.obtener_interacciones_gui('./uploads/'+nombre_archivo_ratings, separador_ratings, encoding_ratings)
			# Se entrena el modelo de Spotlight
			sistema.entrenar_modelo_gui()
		return redirect(url_for('home'))


@app.route("/<path:path>/datasets_prueba", methods=['GET','POST'])
def datasets_prueba(path):
	"""
    Método datasets_prueba. Muestra la página donde seleccionar el dataset cargado que se quiere utilizar.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    """
  	
	global sistema, modelo_clasico_dl, timestamps
	
	# Se genera el formulario
	form = DatasetsPruebaForm(request.form)

	if request.method == 'GET':
		return render_template('datasets_prueba.html', titulo='Datasets Prueba', form=form)
	else:
		if modelo_clasico_dl == 1:
			# Se obtienen los modelos de LightFM
			if form.menu.data == '1':
				# Se obtienen las rutas de las matrices de Movielens 
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_movielens.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_movielens.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_movielens.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_movielens.pickle')
				# Se cargan las matrices de Movielens
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				# Se entrena el modelo de Movielens
				sistema.entrenar_modelo_gui()
			elif form.menu.data == '2':
				# Se obtienen las rutas de las matrices de Anime
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_anime.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_anime.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_anime.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_anime.pickle')
				# Se cargan las matrices de Anime
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				# Se entrena el modelo de Anime
				sistema.entrenar_modelo_gui()
			elif form.menu.data == '3':
				# Se obtienen las rutas de las matrices de Books Crossing
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_books.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_books.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_books.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_books.pickle')
				# Se cargan las matrices de Books Crossing
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				# Se entrena el modelo de Books Crossing
				sistema.entrenar_modelo_gui()
			elif form.menu.data == '4':
				# Se obtienen las rutas de las matrices de LastFM
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_last.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_last.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_last.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_last.pickle')
				# Se cargan las matrices de LastFM
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				# Se entrena el modelo de LastFM
				sistema.entrenar_modelo_gui()
			elif form.menu.data == '5':
				# Se obtienen las rutas de las matrices de Dating Agency
				ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_clasico_dating.pickle')
				ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_clasico_dating.pickle')
				ruta_items = os.path.join(app.config['UPLOAD_FOLDER'], 'item_features_dating.pickle')
				ruta_users = os.path.join(app.config['UPLOAD_FOLDER'], 'user_features_dating.pickle')
				# Se cargan las matrices de Dating Agency
				sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)
				# Se entrena el modelo de Dating Agency
				sistema.entrenar_modelo_gui()
			else:
				# Se cargan las matrices de otro dataset
				sistema.cargar_otras_matrices_gui()	
				# Se entrena el modelo de otro dataset
				sistema.entrenar_modelo_gui()	
		else:
			# Se obtienen los modelos de Spotlight
			if timestamps == 1:
				if form.menu.data == '1':
					# Se obtienen las rutas de las interacciones de Movielens 
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_movielens.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_movielens.pickle')
					# Se cargan las interacciones de Movielens 
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de Movielens
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '2':
					# Se obtienen las rutas de las interacciones de Anime  
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_anime.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_anime.pickle')
					# Se cargan las interacciones de Anime
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de Anime
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '3':
					# Se obtienen las rutas de las interacciones de Books Crossing 
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_books.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_books.pickle')
					# Se cargan las interacciones de Books Crossing 
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de Books Crossing 
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '4':
					# Se obtienen las rutas de las interacciones de LastFM
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_last.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_last.pickle')
					# Se cargan las interacciones de LastFM
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de LastFM
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '5':
					# Se obtienen las rutas de las interacciones de Dating Agency
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_time_dating.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_time_dating.pickle')
					# Se cargan las interacciones de Dating Agency
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de Dating Agency
					sistema.entrenar_modelo_gui()
				else:
					# Se cargan las interacciones de otro dataset
					sistema.cargar_otras_interacciones_gui()
					# Se entrena el modelo de otro dataset	
					sistema.entrenar_modelo_gui()	
			else:
				if form.menu.data == '1':
					# Se obtienen las rutas de las interacciones de Movielens 
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_movielens.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_movielens.pickle')
					# Se cargan las interacciones de Movielens
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de Movielens
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '2':
					# Se obtienen las rutas de las interacciones de Anime 
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_anime.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_anime.pickle')
					# Se cargan las interacciones de Anime
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de Anime
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '3':
					# Se obtienen las rutas de las interacciones de Books Crossing 
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_books.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_books.pickle')
					# Se cargan las interacciones de Books Crossing 
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de Books Crossing 
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '4':
					# Se obtienen las rutas de las interacciones de LastFM
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_last.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_last.pickle')
					# Se cargan las interacciones de LastFM
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de LastFM
					sistema.entrenar_modelo_gui()
				elif form.menu.data == '5':
					# Se obtienen las rutas de las interacciones de Dating Agency
					ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], 'train_dl_dating.pickle')
					ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], 'test_dl_dating.pickle')
					# Se cargan las interacciones de Dating Agency
					sistema.cargar_interacciones_gui(ruta_train, ruta_test)
					# Se entrena el modelo de Dating Agency
					sistema.entrenar_modelo_gui()
				else:
					# Se cargan las matrices de otro dataset
					sistema.cargar_otras_interacciones_gui()	
					# Se entrena el modelo de otro dataset
					sistema.entrenar_modelo_gui()
		return redirect(url_for('home'))


@app.route("/<path:path>/cargar_modelo_clasico", methods=['GET','POST'])
def cargar_modelo_clasico(path):
	"""
    Método cargar_modelo_clasico. Muestra la página donde seleccionar el modelo de LightFM que se quiere cargar.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    """
  	
	global sistema
	
	# Se genera el formulario
	form = CargarModeloClasicoForm(request.form)

	if request.method == 'GET':
		return render_template('cargar_modelo_clasico.html', titulo='Cargar Modelo Clásico', form=form)
	else:
		# Se obtienen las rutas de los modelos y de las matrices que se necesitan
		modelo = request.files['modelo']
		nombre_modelo = secure_filename(modelo.filename)
		ruta_modelo = os.path.join(app.config['UPLOAD_FOLDER'], nombre_modelo)
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

		# Se generan los sistemas
		if '_colab_' in nombre_modelo:
			sistema = SistemaLightFM.SistemaLightFM(opcion_modelo=1)
		elif '_hibrido_' in nombre_modelo:
			sistema = SistemaLightFM.SistemaLightFM(opcion_modelo=2)
		else:
			sistema = SistemaLightFM.SistemaLightFM(opcion_modelo=3)

		# Se carga el modelo
		sistema.cargar_modelo_gui(ruta_modelo)
		# Se cargan las matrices
		sistema.cargar_matrices_gui(ruta_train, ruta_test, ruta_items, ruta_users)

		if form.menu.data == '1':
			return redirect(url_for('ver_metricas', path=path+'/cargar_modelo_clasico'))
		else:
			max_id = sistema.obtener_id_maximo()
			return redirect(url_for('elegir_usuario', path=path+'/cargar_modelo_clasico', max_id=max_id))


@app.route("/<path:path>/cargar_modelo_dl", methods=['GET','POST'])
def cargar_modelo_dl(path):
	"""
    Método cargar_modelo_dl. Muestra la página donde seleccionar el modelo de Spotlight que se quiere cargar.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    """
  	
	global sistema
	
	# Se genera el formulario
	form = CargarModeloDLForm(request.form)

	if request.method == 'GET':
		return render_template('cargar_modelo_dl.html', titulo='Cargar Modelo DL', form=form)
	else:
		# Se obtienen las rutas de los modelos y de las interacciones que se necesitan
		modelo = request.files['modelo']
		nombre_modelo = secure_filename(modelo.filename)
		ruta_modelo = os.path.join(app.config['UPLOAD_FOLDER'], nombre_modelo)
		train = request.files['archivo_train']
		nombre_train = secure_filename(train.filename)
		ruta_train = os.path.join(app.config['UPLOAD_FOLDER'], nombre_train)
		test = request.files['archivo_test']
		nombre_test = secure_filename(test.filename)
		ruta_test = os.path.join(app.config['UPLOAD_FOLDER'], nombre_test)

		# Se generan los sistemas
		if '_expl_' in nombre_modelo:
			sistema = SistemaSpotlight.SistemaSpotlight(opcion_modelo=1)
		elif '_fact_impl_' in nombre_modelo:
			sistema = SistemaSpotlight.SistemaSpotlight(opcion_modelo=2)
		else:
			sistema = SistemaSpotlight.SistemaSpotlight(opcion_modelo=3)

		# Se carga el modelo
		sistema.cargar_modelo_gui(ruta_modelo)
		# Se cargan las interacciones
		sistema.cargar_interacciones_gui(ruta_train, ruta_test)

		if form.menu.data == '1' or '_secuencia_' in nombre_modelo:
			return redirect(url_for('ver_metricas', path=path+'/cargar_modelo_dl'))
		else:
			max_id = sistema.obtener_id_maximo()
			return redirect(url_for('elegir_usuario', path=path+'/cargar_modelo_dl', max_id=max_id))


@app.route("/<path:path>/ver_metricas", methods=['GET','POST'])
def ver_metricas(path):
	"""
    Método ver_metricas. Muestra la página donde ver las métricas y los datos del modelo escogido.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    """
  	
	global sistema, modelo_clasico_dl

	# Se genera el formulario
	form = MetricasForm(request.form)

	# Se obtienen las métricas del modelo
	metricas = sistema.obtener_metricas_gui()

	# Se obtienen los datos del modelo
	datos = sistema.obtener_datos_conjunto_gui()

	if request.method == 'GET':
		return render_template('ver_metricas.html', titulo='Métricas', form=form, metricas=metricas, datos=datos)
	else:
		return redirect(url_for('home'))


@app.route("/<path:path>/<int:max_id>/elegir_usuario", methods=['GET','POST'])
def elegir_usuario(path, max_id):
	"""
    Método elegir_usuario. Muestra la página donde ver elegir el id del usuario cuyas predicciones se quieren calcular.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    max_id: int
    	id del último usuario del dataset
    """
  	
	# Se genera el formulario
	form = ElegirUsuarioForm(request.form)

	# Se genera un mensaje indicando el id máximo
	mensaje = 'Hay ' + str(max_id) + ' usuarios'

	if request.method == 'GET':
		return render_template('elegir_usuario.html', titulo='Elegir Usuario', form=form, mensaje=mensaje)
	else:
		usuario = int(request.form['usuario'])
		return redirect(url_for('ver_predicciones', path=path+'/elegir_usuario', usuario=usuario))


@app.route("/<path:path>/<int:usuario>/ver_predicciones", methods=['GET','POST'])
def ver_predicciones(path, usuario):
	"""
    Método ver_predicciones. Muestra la página donde ver las predicciones del usuario escogido.

    Este método solo se utiliza en la interfaz web.

    Parameters
    ----------

    path: path
    	ruta de la página anterior
    usuario: int
    	id usuario cuyas predicciones se quiren ver
    """
  	
	global sistema

	# Se genera el formulario
	form = PrediccionesForm(request.form)

	# Se obtienen las predicciones del usuario
	predicciones = sistema.obtener_predicciones(usuario)

	if request.method == 'GET':
		return render_template('ver_predicciones.html', titulo='Predicciones', form=form, predicciones=predicciones)
	else:
		return redirect(url_for('home'))


@app.route("/home/añadir_valoraciones", methods=['GET','POST'])
def anadir_valoraciones():
	"""
    Método anadir_valoraciones. Muestra la página donde ver las predicciones del usuario escogido.

    Este método solo se utiliza en la interfaz web.

    No es funcional.
    """
  	
	# Se genera el formulario
	form = AnadirValoracionesForm(request.form)

	if request.method == 'GET':
		return render_template('anadir_valoraciones.html', titulo='Añadir Valoraciones', form=form)
	else:
		return redirect(url_for('home'))


