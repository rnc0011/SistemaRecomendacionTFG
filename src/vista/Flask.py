from flask import Flask, render_template
from Forms import HomeForm

app = Flask(__name__)
app.secret_key = 'development key'

@app.route("/home", methods=['GET','POST'])
def home():
	form = HomeForm()
	return render_template('home.html', titulo='Página Principal', form=form)

if __name__ == '__main__':
	app.run(debug=True)


