from flask import Flask, render_template
from Forms import FormMenu

app = Flask(__name__)
app.secret_key = 'development key'

@app.route("/menu", methods=['GET','POST'])
def menu():
	form = FormMenu()
	return render_template('menu.html', form = form)

if __name__ == '__main__':
	app.run(debug=True)


