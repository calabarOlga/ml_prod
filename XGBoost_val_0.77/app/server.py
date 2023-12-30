import json
import joblib
import pandas as pd
from process_func import *
from flask import Flask, request, render_template, jsonify


# Чтение категорий из файлов CSV
categories_df = pd.read_csv('cats/categories.csv')
last_categories_df = pd.read_csv('cats/last_categories.csv')

# Преобразование DataFrame обратно в список
all_categories = categories_df['Category'].tolist()
unique_last_cats = last_categories_df['Category'].tolist()


# Функция для загрузки модели
def load_model(model_path):
    return joblib.load(model_path)


# Функция для обработки входных данных и выполнения предсказания
def predict_model(input_json, model):
    # Преобразования JSON
    merge_df_test = get_df(input_json, target=False)
    X_test = preprocessing_df(merge_df_test, all_categories, unique_last_cats, target=False)

    # Выполнение предсказания
    prediction = model.predict(X_test)
    y_pred_label = ['female' if pred == 1 else 'male' for pred in prediction][0]

    # Возвращаем предсказание в виде строки
    return y_pred_label

# Создание Flask-приложения
app = Flask(__name__)

model = load_model('model/trained_pipeline.pkl')  # Путь к модели

# Главная страница для загрузки JSON-файла
@app.route('/')
def home():
    return render_template('index.html')

# Роут для обработки POST-запроса с JSON-файлом
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Получение файла из запроса
        file = request.files.get('file')
        if not file:
            # Если файл не был отправлен
            return render_template('index.html', prediction_text='Please, upload a JSON file.')

        try:
            data = json.load(file)
            prediction = predict_model(data, model)
            return render_template('index.html', prediction_text='Gender: {}'.format(prediction))
        except json.JSONDecodeError:
            # Ошибка декодирования JSON
            return render_template('index.html', prediction_text='Invalid JSON file. Please upload a valid JSON file.')
        except Exception as e:
            # Логирование неизвестных ошибок
            print("An error occurred: {}".format(e))
            return render_template('index.html', prediction_text='An error occurred. Please try again.')

    else:
        return jsonify({'status': 'error', 'message': 'Invalid request'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
