from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from wtforms.validators import DataRequired, NumberRange, NoneOf
from wtforms import SelectField, StringField, SubmitField, DecimalField, IntegerField

from ensembles import RandomForestMSE, GradientBoostingMSE


RFOREST_TYPE = "Случайный лес"
BOOSTING_TYPE = "Градиентный бустинг"

RFOREST_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": -1,
    "feature_subsample_size": 0.33,
}

BST_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "feature_subsample_size": 0.33,
    "learning_rate": 0.1
}

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)


class Model:
    def __init__(self):
        self.type = 0
        self.model = None
        self.params = {}

model = Model()


class Validators:
    MAX_N_ESTIMATORS = 10_000
    MAX_MAX_DEPTH = 10_000
    MAX_N_FEAT = 10_000
    MAX_LRATE = 2
    n_estimators = [
        NumberRange(1, MAX_N_ESTIMATORS, f"Параметр \"n_estimators\" должен быть между 1 и {MAX_N_ESTIMATORS}")
    ]
    max_depth = [
        NumberRange(-1, MAX_MAX_DEPTH, f"Параметр \"max_depth\" должен быть между -1 и {MAX_MAX_DEPTH}"),
        NoneOf([0], "Параметр \"max_depth\" не может быть равен нулю")
    ]
    feature_subsample_size = [
        NumberRange(0, MAX_N_FEAT, f"Параметр \"feature_subsample_size\" должен быть между 0 и {MAX_N_FEAT}"),
        NoneOf([0], "Параметр \"feature_subsample_size\" не может быть равен нулю")
    ]
    learning_rate = [
        NumberRange(0, MAX_LRATE, f"Параметр \"learning_rate\" должен быть между 0 и {MAX_LRATE}"),
        NoneOf([0], "Параметр \"learning_rate\" не может быть равен нулю")
    ]


class ModelDropdown(FlaskForm):
    dropdown = SelectField("", choices=[(RFOREST_TYPE, "Случайный лес"),
                                        (BOOSTING_TYPE, "Градиентный бустинг")])
    submit = SubmitField("Выбрать")


class RForestParamsSelectionForm(FlaskForm):
    n_estimators_field = IntegerField("n_estimators", default=RFOREST_DEFAULT_PARAMS["n_estimators"],
                                      validators=Validators.n_estimators)
    max_depth_field = IntegerField("max_depth", default=RFOREST_DEFAULT_PARAMS["max_depth"],
                                   validators=Validators.max_depth)
    feature_subsample_size_field = DecimalField("feature_subsample_size",
                                                default=RFOREST_DEFAULT_PARAMS["feature_subsample_size"],
                                                validators=Validators.feature_subsample_size)
    learning_rate_field = None
    submit = SubmitField("Выбрать")


class BstParamsSelectionForm(FlaskForm):
    n_estimators_field = IntegerField("n_estimators", default=BST_DEFAULT_PARAMS["n_estimators"],
                                      validators=Validators.n_estimators)
    max_depth_field = IntegerField("max_depth", default=BST_DEFAULT_PARAMS["max_depth"],
                                   validators=Validators.max_depth)
    feature_subsample_size_field = DecimalField("feature_subsample_size",
                                                default=BST_DEFAULT_PARAMS["feature_subsample_size"],
                                                validators=Validators.feature_subsample_size)
    learning_rate_field = DecimalField("learning_rate", default=BST_DEFAULT_PARAMS["learning_rate"],
                                       validators=Validators.learning_rate)
    submit = SubmitField("Выбрать")


class GotoOriginForm(FlaskForm):
    submit = SubmitField("Вернуться в начало")


@app.route("/", methods=["GET", "POST"])
def choose_model():
    global model
    model = Model()

    model_dropdown = ModelDropdown()

    if model_dropdown.validate_on_submit():
        model.type = model_dropdown.dropdown.data
        return redirect(url_for("params_selection"))

    return render_template("choose_model.html", form=model_dropdown)


@app.route("/params_selection", methods=["GET", "POST"])
def params_selection():
    if model.type == BOOSTING_TYPE:
        params_selection_form = BstParamsSelectionForm()
    else:
        params_selection_form = RForestParamsSelectionForm()

    goto_origin_form = GotoOriginForm()
    
    if params_selection_form.validate_on_submit() and request.form["submit"] == "Выбрать":
        model.params["n_estimators"] = params_selection_form.n_estimators_field.data
        model.params["max_depth"] = params_selection_form.max_depth_field.data
        model.params["feature_subsample_size"] = float(params_selection_form.feature_subsample_size_field.data)
        model.params["learning_rate"] = (None
                                        if params_selection_form.learning_rate_field is None
                                        else float(params_selection_form.learning_rate_field.data))
        
        return redirect(url_for("fit_model"))
    
    if goto_origin_form.validate_on_submit() and request.form["submit"] == "Вернуться в начало":
        return redirect(url_for("choose_model"))
    
    return render_template("params_selection.html", model_type=model.type,
                           form1=params_selection_form,
                           form2=goto_origin_form)


@app.route("/fit_model", methods=["GET", "POST"])
def fit_model():
    app.logger.info(model.type)
    app.logger.info(str(model.params))
    return render_template("fit_model.html")
