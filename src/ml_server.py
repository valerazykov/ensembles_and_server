from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired, NumberRange, NoneOf
from wtforms import SelectField, StringField, SubmitField, DecimalField, IntegerField, FileField

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

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

    def fit(self, X, y, X_val=None, y_val=None):
        if self.type == RFOREST_TYPE:
            self.model = RandomForestMSE(**self.params)
        elif self.type == BOOSTING_TYPE:
            self.model = GradientBoostingMSE(**self.params)
        else:
            raise TypeError("неизвестный тип модели")
        
        self.model.fit(X, y, X_val, y_val)
        return self
    
    def predict(self, X):
        return self.model.predict(X)


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


class FitPageForm(FlaskForm):
    file_train_path = FileField('Файл для обучения', validators=[
        #DataRequired('Укажите файл'),
        FileAllowed(['csv'], 'Поддерживается только CSV формат')
    ])
    file_val_path = FileField('Файл для валидиции (опционально)', validators=[
        FileAllowed(['csv'], 'Поддерживается только CSV формат')
    ])
    target_field = StringField("Название колонки с целевой переменной", validators=[
        #DataRequired("Введине название колонки с целевой переменной")
    ])
    submit_open = SubmitField("Открыть файлы и обучить модель")
    submit_back = SubmitField("Назад")
    submit_goto_origin = SubmitField("Вернуться в начало")



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
    if model.type not in {RFOREST_TYPE, BOOSTING_TYPE}:
        return goto_origin()
    
    title = "Выбор параметров модели"
    header = f"Выберите параметры модели \"{model.type}\""

    if model.type == BOOSTING_TYPE:
        params_selection_form = BstParamsSelectionForm()
    else:
        params_selection_form = RForestParamsSelectionForm()
    
    if params_selection_form.validate_on_submit():
        if params_selection_form.submit_select.data:
            model.params["n_estimators"] = params_selection_form.n_estimators_field.data
            if params_selection_form.max_depth_field.data != -1:
                model.params["max_depth"] = params_selection_form.max_depth_field.data
            model.params["feature_subsample_size"] = float(params_selection_form.feature_subsample_size_field.data)
            if params_selection_form.learning_rate_field is not None:
                model.params["learning_rate"] = float(params_selection_form.learning_rate_field.data)
            
            return redirect(url_for("fit_model"))
        else:
            return goto_origin()
    
    return render_template("from_form.html", title=title, header=header,
                           form=params_selection_form)


@app.route("/fit_model", methods=["GET", "POST"])
@app.route("/fit_model/<string:message>", methods=["GET", "POST"])
def fit_model(message=None):
    if model.type not in {RFOREST_TYPE, BOOSTING_TYPE}:
        return goto_origin()

    title = "Обучение модели"
    if message is None:
        header = f"Выберите файлы для обучения модели \"{model.type}\""
    else:
        header = message
    fit_form = FitPageForm()

    if request.method == 'POST':
        if fit_form.validate_on_submit() and fit_form.submit_open.data:
            if not fit_form.file_train_path.data:
                return redirect("/fit_model/Необходимо передать данные для обучения")
            if not fit_form.target_field.data:
                return redirect("/fit_model/Необходимо указать название колонки с целевой переменной")
            
            try:
                data_train = pd.read_csv(fit_form.file_train_path.data)
            except EmptyDataError:
                return redirect("/fit_model/Необходимо передать непустой файл для обучения")
            
            # use only numeric features
            data_train = data_train.select_dtypes(include=[np.number])

            if data_train.shape[0] == 0 or data_train.shape[1] <= 1:
                return redirect("/fit_model/Передайте корректный файл для обучения")
            
            if fit_form.target_field.data not in data_train.columns:
                return redirect("/fit_model/В файле для обучения нет указанной колонки с целевой переменной")

            X_train = data_train.drop(columns=[fit_form.target_field.data]).to_numpy()
            y_train = data_train[fit_form.target_field.data].to_numpy()


            if fit_form.file_val_path.data:
                try:
                    data_val = pd.read_csv(fit_form.file_val_path.data)
                except:
                    return redirect("/fit_model/Файл для валидации не может быть пустым")
                
                data_val = data_val.select_dtypes(include=[np.number])
                
                if data_val.shape[0] == 0:
                    return redirect("/fit_model/Файл для валидации должен содержать хотя бы один объект")

                if data_val.shape[1] != data_train.shape[1] or (data_val.columns != data_train.columns).any():
                    return redirect("/fit_model/Файл для валидации должен быть согласован с файлом для обучения")
                
                X_val = data_val.drop(columns=[fit_form.target_field.data]).to_numpy()
                y_val = data_val[fit_form.target_field.data].to_numpy()
            else:
                X_val = None
                y_val = None

            model.fit(X_train, y_train, X_val, y_val)

            return redirect(url_for("predict"))
        
        if fit_form.submit_back.data:
            model.params = {}
            return redirect(url_for("params_selection"))
        if fit_form.submit_goto_origin.data:
            return goto_origin()

    return render_template("from_form.html", title=title, header=header, form=fit_form)


@app.route("/predict")
def predict():
    return "<h3>predict</h3>"
