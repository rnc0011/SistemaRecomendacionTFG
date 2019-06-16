from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired

class HomeForm(FlaskForm):
	choices = [(1, 'Construir modelo de recomendación'),(2, 'Cargar modelo'),(3, 'Añadir valoraciones a conjunto existente')]
	menu = RadioField(choices=choices)
	submit = SubmitField('Enviar')
