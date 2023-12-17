from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect, send_from_directory

from flask_wtf.file import FileAllowed
from wtforms.validators import NumberRange, NoneOf
from wtforms import SelectField, StringField, SubmitField, DecimalField, IntegerField, FileField

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        self.feat_columns = []
        self.target_name = ""
        self.train_data_df = None

    def fit(self, X, y, X_val=None, y_val=None):
        if self.type == RFOREST_TYPE:
            self.model = RandomForestMSE(**self.params)
        elif self.type == BOOSTING_TYPE:
            self.model = GradientBoostingMSE(**self.params)
        else:
            raise TypeError("неизвестный тип модели")

        self.model.fit(X, y, X_val, y_val, need_train_errors_history=True)
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
    file_train_path = FileField('CSV-файл для обучения', validators=[
        FileAllowed(['csv'], 'Поддерживается только CSV формат')
    ])
    file_val_path = FileField('CSV-файл для валидиции (опционально)', validators=[
        FileAllowed(['csv'], 'Поддерживается только CSV формат')
    ])
    target_field = StringField("Название колонки с целевой переменной", validators=[
    ])
    submit_open = SubmitField("Открыть файлы и обучить модель")
    submit_back = SubmitField("Назад")
    submit_goto_origin = SubmitField("Вернуться в начало")


class PredictForm(FlaskForm):
    file_predict_path = FileField('CSV-файл для предсказания', validators=[
        FileAllowed(['csv'], 'Поддерживается только CSV формат')
    ])

    submit_pred = SubmitField("Сделать и скачать предсказание")
    submit_back = SubmitField("Назад")
    submit_goto_origin = SubmitField("Вернуться в начало")
    submit_info = SubmitField("Посмотреть информацию о модели")


class ModelInfoForm(FlaskForm):
    submit_back = SubmitField("Назад")


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

    try:
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
    except Exception:
        return goto_origin()


@app.route("/fit_model", methods=["GET", "POST"])
@app.route("/fit_model/<string:message>", methods=["GET", "POST"])
def fit_model(message=None):
    if model.type not in {RFOREST_TYPE, BOOSTING_TYPE}:
        return goto_origin()

    try:
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
                    except EmptyDataError:
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
                model.feat_columns = data_train.columns.drop(fit_form.target_field.data)
                model.train_data_df = data_train
                model.target_name = fit_form.target_field.data

                return redirect(url_for("predict"))

            if fit_form.submit_back.data:
                model.params = {}
                return redirect(url_for("params_selection"))
            if fit_form.submit_goto_origin.data:
                return goto_origin()

        return render_template("from_form.html", title=title, header=header, form=fit_form)
    except Exception:
        return goto_origin()


@app.route("/predict", methods=["GET", "POST"])
@app.route("/predict/<string:message>", methods=["GET", "POST"])
def predict(message=None):
    if model.type not in {RFOREST_TYPE, BOOSTING_TYPE}:
        return goto_origin()

    try:
        title = "Предсказание"
        if message is None:
            header = "Выберите файл для предсказания"
        else:
            header = message
        form = PredictForm()

        if form.validate_on_submit():
            if form.submit_back.data:
                return redirect(url_for("fit_model"))
            if form.submit_goto_origin.data:
                return goto_origin()
            if form.submit_info.data:
                return redirect(url_for("model_info"))
            if form.submit_pred.data:
                if not form.file_predict_path.data:
                    return redirect("/predict/Необходимо передать данные для предсказания")
                try:
                    X_test = pd.read_csv(form.file_predict_path.data)
                except EmptyDataError:
                    return redirect("/predict/Файл для предсказания должен быть не пустым")

                X_test = X_test.select_dtypes(include=[np.number])

                if X_test.shape[0] == 0:
                    return redirect("/predict/Файл для предсказания должен содержать хотя бы один объект")

                if X_test.shape[1] == model.feat_columns.shape[0] + 1:
                    if model.target_name not in X_test.columns:
                        return redirect("/predict/Файл для предсказания должен быть согласован с файлом для обучения")
                    X_test = X_test.drop(columns=[model.target_name])

                if (X_test.columns != model.feat_columns).any():
                    return redirect("/predict/Файл для предсказания должен быть согласован с файлом для обучения")

                X_test = X_test.to_numpy()
                pred = model.predict(X_test)
                np.savetxt("pred.txt", pred)

                return redirect("/upload/pred.txt")

        return render_template("from_form.html", form=form, title=title, header=header)
    except Exception:
        return goto_origin()


@app.route("/model_info", methods=["POST", "GET"])
def model_info():
    if model.type not in {RFOREST_TYPE, BOOSTING_TYPE}:
        return goto_origin()

    try:
        title = "Информация о модели"
        header = f"Информация о модели \"{model.type}\""
        form = ModelInfoForm()

        if form.validate_on_submit():
            return redirect(url_for("predict"))

        x = np.arange(1, model.params["n_estimators"] + 1)
        train_errors_history = model.model.train_errors_history
        val_errors_history = model.model.ensemble_errors_history

        df_train = pd.DataFrame({
            "число деревьев": x,
            "RMSE": train_errors_history
        })

        if val_errors_history is not None:
            fig = make_subplots(rows=1, cols=2, x_title="число деревьев", y_title="RMSE")

            df_val = pd.DataFrame({
                "число деревьев": x,
                "RMSE": val_errors_history
            })

            fig.add_trace(go.Bar(x=df_train["число деревьев"], y=df_train["RMSE"], name="Обучающая выборка"), 1, 1)

            fig.add_trace(go.Bar(x=df_val["число деревьев"], y=df_val["RMSE"], name="Валидационная выборка"), 1, 2)

            fig.update_layout(title_text="График зависимости RMSE от числа деревьев", title_x=0.5)

        else:
            fig = px.bar(df_train, x="число деревьев", y="RMSE",
                         title="График зависимости RMSE на обучающей выборке от числа деревьев")
            fig.update_layout(title_x=0.5)

        # Create graphJSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        params_to_display = model.params.copy()
        if "learning_rate" not in params_to_display:
            params_to_display["learning_rate"] = None
        if "max_depth" not in params_to_display:
            params_to_display["max_depth"] = "-1"

        params_to_display["n_sam"] = model.train_data_df.shape[0]
        params_to_display["n_feat"] = model.train_data_df.shape[1] - 1

        # Use render_template to pass graphJSON to html
        return render_template('model_info.html', graphJSON=graphJSON,
                               title=title, header=header, form=form,
                               **params_to_display)
    except Exception:
        return goto_origin()


@app.route("/upload/<path:name>")
def upload(name):
    try:
        return send_from_directory(".", name, as_attachment=True)
    except Exception:
        return goto_origin()
