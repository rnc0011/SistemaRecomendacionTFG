from flask_wtf import Form
from wtforms import RadioField

class FormMenu(Form):
	dataset = RadioField('Conjunto de datos', choices=[(1, 'Movilens'),(2, 'Anime'),(3, 'Book Crossing'),(4, 'LastFM'),(5, 'Dating Agency')])