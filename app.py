from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Загрузка модели
model = joblib.load('best_random_forest.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        datetime_str = data.get('datetime', '')
        
        # Преобразование строки в datetime
        dt = pd.to_datetime(datetime_str)
        
        # Подготовка признаков как при обучении
        features = prepare_features(dt)
        
        # Получение предсказания от модели
        prediction = model.predict([features])[0]
        
        return jsonify({
            'success': True,
            'prediction': {
                'datetime': datetime_str,
                'predicted_orders': round(float(prediction), 2)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def prepare_features(dt):
    """Подготовка признаков для модели"""
    features = {
        'hour': dt.hour,
        'day': dt.day,
        'dayofweek': dt.dayofweek,
    }
    
    # Добавляем лаги и rolling_mean как нули, 
    # так как у нас нет исторических данных в реальном времени
    for lag in range(1, 25):  # 24 лага
        features[f'lag_{lag}'] = 0
    
    features['rolling_mean'] = 0
    
    # Преобразование в список в том же порядке, что и при обучении
    feature_names = ['hour', 'day', 'dayofweek'] + \
                   [f'lag_{i}' for i in range(1, 25)] + \
                   ['rolling_mean']
    
    return [features[name] for name in feature_names]

if __name__ == '__main__':
    app.run(debug=True) 