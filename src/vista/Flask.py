from flask import Flask, render_template, flash, request
from Forms import HomeForm

app = Flask(__name__)
app.secret_key = 'development key'

@app.route("/home", methods=['POST'])
def home():
	form = HomeForm(request.form)
	print(form.menu.data)
	if form.menu.data == 1:
		print("hola")
		return redirect(url_for('construir_modelo'))
	return render_template('home.html', titulo='Página Principal', form=form)

@app.route("/construir_modelo", methods=['GET','POST'])
def construir_modelo():
	return render_template('construir_modelo.html', titulo='Construir Modelo')

if __name__ == '__main__':
	app.run(debug=True)


