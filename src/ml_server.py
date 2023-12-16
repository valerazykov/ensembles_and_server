from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from wtforms.validators import DataRequired
from wtforms import SelectField, StringField, SubmitField

from ensembles import RandomForestMSE, GradientBoostingMSE


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)

class Model:
    def __init__(self):
        self.type = ""
        self.model = None

model = Model()


class ModelDropdown(FlaskForm):
    dropdown = SelectField("", choices=[("Случайный лес", "Случайный лес"),
                                        ("Градиентный бустинг", "Градиентный бустинг")])
    submit = SubmitField("Выбрать")

@app.route("/", methods=["GET", "POST"])
def choose_model():
    model_dropdown = ModelDropdown()

    if model_dropdown.validate_on_submit():
        model.type = model_dropdown.dropdown.data
        return redirect(url_for("params_selection"))

    return render_template("choose_model.html", form=model_dropdown)

@app.route("/params_selection", methods=["GET", "POST"])
def params_selection():
    return render_template("params_selection.html", model_type=model.type)
