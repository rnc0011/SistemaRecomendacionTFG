from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, SelectField, MultipleFileField, IntegerField, FloatField, FileField

class HomeForm(FlaskForm):
	choices = [(1, 'Construir modelo de recomendación'), (2, 'Cargar modelo'), (3, 'Añadir valoraciones a conjunto existente')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class DatasetForm(FlaskForm):
	choices = [(1, 'Con un nuevo dataset'), (2, 'Con datasets de prueba')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class NuevoDatasetForm(FlaskForm):
	encoding = SelectField(choices=[('utf-8', 'utf-8'), ('ISO-8859-1', 'ISO-8859-1')])
	separador = SelectField(choices=[('|', '|'), ('\t', '\t'), (',', ','), (';', ';')])
	archivo = FileField('Archivo .csv')
	submit = SubmitField('Siguiente')
	mas_archivos = SubmitField('Más archivos')


class DatasetsPruebaForm(FlaskForm):
	choices = [(1, 'Movielens'), (2, 'Anime'), (3, 'Book Crossing'), (4, 'Last FM'), (5, 'Dating Agency'), (6, 'Otro')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class ElegirModeloForm(FlaskForm):
	choices = [(1, 'Modelo Clásico'), (2, 'Modelo basado en Aprendizaje Profundo')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class ElegirModeloClasicoForm(FlaskForm):
	choices = [(1, 'Modelo Colaborativo'), (2, 'Modelo Híbrido'), (3, 'Modelo por Contenido')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class ElegirModeloDLForm(FlaskForm):
	choices = [(1, 'Modelo de Factorización Explícito'), (2, 'Modelo de Factorización Implícito'), (3, 'Modelo de Secuencia Implícito')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')


class ParamClasicoForm(FlaskForm):
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
	loss = SelectField(choices=[('regression', 'Regression (Fact. Explícito)'), ('poisson', 'Poisson (Fact. Explícito)'), ('logistic', 'logistic (Fact. Explícito)'), ('pointwise', 'Pointwise (Fact. Implícito y Secuencia)'), ('bpr', 'BPR (Fact. Implícito y Secuencia)'), ('hinge', 'Hinge (Fact. Implícito y Secuencia)'), ('adaptive hinge', 'Adaptive Hinge (Fact. Implícito y Secuencia)')])
	embedding_dim = IntegerField(default=32)
	n_iter = IntegerField(default=10)
	batch_size = IntegerField(default=256)
	l2 = FloatField(default=0.0)
	learning_rate = FloatField(default=0.01)
	epochs = IntegerField(default=30)
	submit = SubmitField('Construir')


class AnadirValoracionesForm(FlaskForm):
	modelo = FileField()
	valoraciones = FileField()
	submit = SubmitField('Añadir')


class CargarModeloForm(FlaskForm):
	modelo = FileField()
	archivos = MultipleFileField('Matrices')
	menu = RadioField(choices=[(1, 'Métricas'), (2, 'Predicciones')])
	submit = SubmitField('Siguiente')


class ElegirUsuarioForm(FlaskForm):
	usuario = IntegerField()
	submit = SubmitField('Siguiente')


class PrediccionesForm(FlaskForm):
	submit = SubmitField('Finalizar')


class PrediccionesForm(FlaskForm):
	submit = SubmitField('Finalizar')


class MetricasForm(FlaskForm):
	submit = SubmitField('Finalizar')


	
