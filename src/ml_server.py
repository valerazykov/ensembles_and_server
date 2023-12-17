from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired, NumberRange, NoneOf
from wtforms import SelectField, StringField, SubmitField, DecimalField, IntegerField, FileField

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
    submit_select = SubmitField("Выбрать")
    submit_goto_origin = SubmitField("Вернуться в начало")


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
    submit_select = SubmitField("Выбрать")
    submit_goto_origin = SubmitField("Вернуться в начало")


class FileForm(FlaskForm):
    file_path = FileField('Path', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Open File')


@app.route("/", methods=["GET", "POST"])
def choose_model():
    model_dropdown = ModelDropdown()
    title = "Выбор модели"
    header = "Выберите модель"

    if model_dropdown.validate_on_submit():
        model.type = model_dropdown.dropdown.data
        return redirect(url_for("params_selection"))

    return render_template("from_form.html", title=title, header=header, form=model_dropdown)

def goto_origin():
    global model
    model = Model()
    return redirect(url_for("choose_model"))

@app.route("/params_selection", methods=["GET", "POST"])
def params_selection():
    title = "Выбор параметров модели"
    header = f"Выберите параметры модели \"{model.type}\""

    if model.type == BOOSTING_TYPE:
        params_selection_form = BstParamsSelectionForm()
    else:
        params_selection_form = RForestParamsSelectionForm()
    
    if params_selection_form.validate_on_submit():
        if params_selection_form.submit_select.data:
            model.params["n_estimators"] = params_selection_form.n_estimators_field.data
            model.params["max_depth"] = params_selection_form.max_depth_field.data
            model.params["feature_subsample_size"] = float(params_selection_form.feature_subsample_size_field.data)
            model.params["learning_rate"] = (None
                                            if params_selection_form.learning_rate_field is None
                                            else float(params_selection_form.learning_rate_field.data))
            
            return redirect(url_for("fit_model"))
        else:
            return goto_origin()
    
    return render_template("from_form.html", title=title, header=header,
                           form=params_selection_form)


@app.route("/fit_model", methods=["GET", "POST"])
def fit_model():
    title = "Обучение модели"
    header = f"Выберите файлы для обучения модели \"{model.type}\""
    file_form = FileForm()
    return render_template("from_form.html", title=title, header=header, form=file_form)
