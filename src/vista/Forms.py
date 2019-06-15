from flask_wtf import Form
from wtforms import RadioField

class HomeForm(Form):
	choices = [(1, 'Construir modelo de recomendación'),(2, 'Cargar modelo'),(3, 'Añadir valoraciones a conjunto existente')]
	menu = RadioField('¿Qué quieres hacer?', choices=choices)

	