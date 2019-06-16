from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, SelectField, MultipleFileField

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
	separador = SelectField(choices=[('\\', '\\'), ('\t', '\t'), (',', ','), (';', ';')])
	archivos = MultipleFileField('Archivos .csv')
	submit = SubmitField('Siguiente')

class DatasetsPruebaForm(FlaskForm):
	choices = [(1, 'Movielens'), (2, 'Anime'), (3, 'Book Crossing'), (4, 'Last FM'), (5, 'Dating Agency'), (6, 'Otro')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Siguiente')



