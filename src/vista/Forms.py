# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:10:36 2019

@author: Raúl Negro Carpintero
"""

# Se importa todo lo necesario
from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, SelectField, MultipleFileField, IntegerField, FloatField, FileField


class HomeForm(FlaskForm):
	"""
	Clase HomeForm. Genera el formulario de la página principal.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    choices: list
    	contiene las opciones del formulario
    menu: RadioField
    	menu de la página
    submit: SubmitField
    	botón de submit
	"""

	choices = [(1, 'Construir modelo de recomendación'), (2, 'Cargar modelo'), (3, 'Añadir valoraciones a conjunto existente')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class DatasetForm(FlaskForm):
	"""
	Clase DatasetForm. Genera el formulario de la página donde escoger entre usar un dataset ya guardado o uno nuevo.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    choices: list
    	contiene las opciones del formulario
    menu: RadioField
    	menu de la página
    submit: SubmitField
    	botón de submit
	"""
	
	choices = [(1, 'Con un nuevo dataset'), (2, 'Con datasets de prueba')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class NuevoDatasetForm(FlaskForm):
	"""
	Clase NuevoDatasetForm. Genera el formulario de la página donde seleccionar los archivos que se quieren utilizar.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    encoding_ratings: SelectField
    	encoding del archivo de valoraciones
    separador_ratings: SelectField
    	separador del archivo de valoraciones
    archivo_ratings: FileField
  		archivo con las valoraciones
  	encoding_users: SelectField
    	encoding del archivo de usuarios
    separador_users: SelectField
    	separador del archivo de usuarios
    archivo_users: FileField
  		archivo con los usuarios
  	encoding_items: SelectField
    	encoding del archivo de ítems
    separador_items: SelectField
    	separador del archivo de ítems
    archivo_items: FileField
  		archivo con los ítems
    submit: SubmitField
    	botón de submit
	"""
	
	encoding_ratings = SelectField(choices=[('utf-8', 'utf-8'), ('ISO-8859-1', 'ISO-8859-1')])
	separador_ratings = SelectField(choices=[('|', '|'), (' ', ' '), (',', ','), (';', ';')])
	archivo_ratings = FileField('Archivo .csv de valoraciones')
	encoding_users = SelectField(choices=[('utf-8', 'utf-8'), ('ISO-8859-1', 'ISO-8859-1')])
	separador_users = SelectField(choices=[('|', '|'), (' ', ' '), (',', ','), (';', ';')])
	archivo_users = FileField('Archivo .csv de usuarios')
	encoding_items = SelectField(choices=[('utf-8', 'utf-8'), ('ISO-8859-1', 'ISO-8859-1')])
	separador_items = SelectField(choices=[('|', '|'), (' ', ' '), (',', ','), (';', ';')])
	archivo_items = FileField('Archivo .csv de items')
	submit = SubmitField('Siguiente')


class DatasetsPruebaForm(FlaskForm):
	"""
	Clase DatasetsPruebaForm. Genera el formulario de la página donde seleccionar el dataset cargado que se quiere utilizar.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    choices: list
    	contiene las opciones del formulario
    menu: RadioField
    	menu de la página
    submit: SubmitField
    	botón de submit
	"""
	
	choices = [(1, 'Movielens'), (2, 'Anime'), (3, 'Book Crossing'), (4, 'Last FM'), (5, 'Dating Agency'), (6, 'Otro')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class ElegirModeloForm(FlaskForm):
	"""
	Clase ElegirModeloForm. Genera el formulario de la página de selección del tipo de modelo.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    choices: list
    	contiene las opciones del formulario
    menu: RadioField
    	menu de la página
    submit: SubmitField
    	botón de submit
	"""
	
	choices = [(1, 'Modelo Clásico'), (2, 'Modelo basado en Aprendizaje Profundo')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class ElegirModeloClasicoForm(FlaskForm):
	"""
	Clase ElegirModeloClasicoForm. Genera el formulario de la página de selección del tipo de modelo clásico.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    choices: list
    	contiene las opciones del formulario
    menu: RadioField
    	menu de la página
    submit: SubmitField
    	botón de submit
	"""
	
	choices = [(1, 'Modelo Colaborativo'), (2, 'Modelo Híbrido'), (3, 'Modelo por Contenido')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class ElegirModeloDLForm(FlaskForm):
	"""
	Clase ElegirModeloDLForm. Genera el formulario de la página de selección del tipo de modelo basado en aprendizaje profundo.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    choices: list
    	contiene las opciones del formulario
    menu: RadioField
    	menu de la página
    submit: SubmitField
    	botón de submit
	"""
	
	choices = [(1, 'Modelo de Factorización Explícito'), (2, 'Modelo de Factorización Implícito'), (3, 'Modelo de Secuencia Implícito')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class TimestampsForm(FlaskForm):
	"""
	Clase TimestampsForm. Genera el formulario de la página donde indicar si el dataset que se va a utilizar lleva timestamps o no.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    choices: list
    	contiene las opciones del formulario
    menu: RadioField
    	menu de la página
    submit: SubmitField
    	botón de submit
	"""
	
	choices = [(1, 'Utilizar timestamps'), (2, 'No utilizar timestamps')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class ParamClasicoForm(FlaskForm):
	"""
	Clase ParamClasicoForm. Genera el formulario de la página donde indicar los parámetros con los que obtener el modelo de LightFM.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    no_components: IntegerField
    	parámetro no_components
	k: IntegerField
    	parámetro k
    n: IntegerField
    	parámetro n
    learning_schedule: SelectField
    	parámetro learning_schedule
    loss: SelectField
    	parámetro loss
    learning_rate: FloatField
    	parámetro learning_rate
    rho: FloatField
    	parámetro rho
    epsilon: FloatField
    	parámetro epsilon
    item_alpha: FloatField
    	parámetro item_alpha
    user_alpha: FloatField
    	parámetro user_alpha
    max_sampled: IntegerField
    	parámetro max_sampled
    epochs: IntegerField
    	parámetro epochs
    submit: SubmitField
    	botón de submit
	"""
	
	no_components = IntegerField(default=10)
	k = IntegerField(default=5)
	n = IntegerField(default=10)
	learning_schedule = SelectField(default='adagrad', choices=[('adagrad', 'adagrad'), ('adadelta', 'adadelta')])
	loss = SelectField(default='logistic',choices=[('logistic', 'logistic'), ('bpr', 'BPR'), ('warp', 'WARP'), ('warp-kos', 'k-OS WARP')])
	learning_rate = FloatField(default=0.05)
	rho = FloatField(default=0.95)
	epsilon = FloatField(default=1e-06)
	item_alpha = FloatField(default=0.0)
	user_alpha = FloatField(default=0.0)
	max_sampled = IntegerField(default=10)
	epochs = IntegerField(default=30)
	submit = SubmitField('Construir')


class ParamDLForm(FlaskForm):
	"""
	Clase ParamDLForm. Genera el formulario de la página donde indicar los parámetros con los que obtener el modelo de Spotlight.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    loss: SelectField
    	parámetro loss
	embedding_dim: IntegerField
    	parámetro embedding_dim
    n_iter: IntegerField
    	parámetro n_iter
    batch_size: IntegerField
    	parámetro batch_size
    l2: FloatField
    	parámetro l2
    learning_rate: FloatField
    	parámetro learning_rate
    representation: SelectField
    	parámetro representation
    submit: SubmitField
    	botón de submit
	"""
	
	loss = SelectField(choices=[('regression', 'Regression (Fact. Explícito)'), ('poisson', 'Poisson (Fact. Explícito)'), ('logistic', 'logistic (Fact. Explícito)'), ('pointwise', 'Pointwise (Fact. Implícito y Secuencia)'), ('bpr', 'BPR (Fact. Implícito y Secuencia)'), ('hinge', 'Hinge (Fact. Implícito y Secuencia)'), ('adaptive hinge', 'Adaptive Hinge (Fact. Implícito y Secuencia)')])
	embedding_dim = IntegerField(default=32)
	n_iter = IntegerField(default=10)
	batch_size = IntegerField(default=256)
	l2 = FloatField(default=0.0)
	learning_rate = FloatField(default=0.01)
	representation = SelectField(choices=[('pooling', 'Pooling'), ('cnn', 'cnn'), ('lstm', 'lstm'), ('mixture', 'mixture')])
	submit = SubmitField('Construir')


class AnadirValoracionesForm(FlaskForm):
	"""
	Clase AnadirValoracionesForm. Genera el formulario de la página donde ver las predicciones del usuario escogido.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    modelo: FileField
    	modelo que se quiere utilizar
	valoraciones: FileField
    	valoraciones que se quieren añadir
    submit: SubmitField
    	botón de submit
	"""
	
	modelo = FileField()
	valoraciones = FileField()
	submit = SubmitField('Añadir')


class CargarModeloClasicoForm(FlaskForm):
	"""
	Clase CargarModeloClasicoForm. Genera el formulario de la página donde seleccionar el modelo de LightFM que se quiere cargar.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    modelo: FileField
    	modelo que se quiere utilizar
	archivo_train: FileField
    	archivo con el conjunto de entrenamiento
    archivo_test: FileField
    	archivo con el conjunto de test
    archivo_users: FileField
		archivo con las features de los usuarios
    archivo_items: FileField
		archivo con las features de los ítems
    menu: RadioField
    	menu de la página
    submit: SubmitField
    	botón de submit
	"""
	
	modelo = FileField('Archivo .pickle con el modelo')
	archivo_train = FileField('Archivo .pickle de entrenamiento')
	archivo_test = FileField('Archivo .pickle de test')
	archivo_users = FileField('Archivo .pickle de usuarios')
	archivo_items = FileField('Archivo .pickle de items')
	menu = RadioField(choices=[(1, 'Métricas'), (2, 'Predicciones')])
	submit = SubmitField('Siguiente')


class CargarModeloDLForm(FlaskForm):
	"""
	Clase CargarModeloDLForm. Genera el formulario de la página donde seleccionar el modelo de Spotlight que se quiere cargar.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    modelo: FileField
    	modelo que se quiere utilizar
	archivo_train: FileField
    	archivo con el conjunto de entrenamiento
    archivo_test: FileField
    	archivo con el conjunto de test
    menu: RadioField
    	menu de la página
    submit: SubmitField
    	botón de submit
	"""
	
	modelo = FileField('Archivo .pickle con el modelo')
	archivo_train = FileField('Archivo .pickle de entrenamiento')
	archivo_test = FileField('Archivo .pickle de test')
	menu = RadioField(choices=[(1, 'Métricas'), (2, 'Predicciones')])
	submit = SubmitField('Siguiente')


class ElegirUsuarioForm(FlaskForm):
	"""
	Clase ElegirUsuarioForm. Genera el formulario de la página donde ver elegir el id del usuario cuyas predicciones se quieren calcular.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    usuario: IntegerField
    	id del usuario cuyas predicciones se quiren obtener
    submit: SubmitField
    	botón de submit
	"""
	
	usuario = IntegerField()
	submit = SubmitField('Siguiente')


class PrediccionesForm(FlaskForm):
	"""
	Clase PrediccionesForm. Genera el formulario de la página donde ver las predicciones del usuario escogido.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    submit: SubmitField
    	botón de submit
	"""
	
	submit = SubmitField('Finalizar')


class MetricasForm(FlaskForm):
	"""
	Clase MetricasForm. Genera el formulario de la página donde ver las métricas y los datos del modelo escogido.

	Parameters
	----------

	form: FlaskForm
		formulario

	Attributes
    ----------

    submit: SubmitField
    	botón de submit
	"""
	
	submit = SubmitField('Finalizar')


	
