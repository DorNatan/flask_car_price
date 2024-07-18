import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
from car_data_prep import prepare_data  # ודא שהפונקציה הזו קיימת בקובץ car_data_prep.py

app = Flask(__name__)

# טעינת המודל המאומן
model = pickle.load(open('trained_model.pkl', 'rb'))

# קריאת קובץ הנתונים 
datafile = "https://raw.githubusercontent.com/DorNatan/flask_car_price/main/dataset.csv"
dataset = pd.read_csv(datafile)

# הכנת הדאטה
prepared_data = prepare_data(dataset)

# הפרדת התכונות (X) והיעד (y)
X = prepared_data.drop(columns=['Price'])
y = prepared_data['Price']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # קבלת הנתונים מהטופס
    form_data = request.form.to_dict()
    data = pd.DataFrame([form_data])

    # הכנת הנתונים באמצעות הפונקציה prepare_data
    prepared_data = prepare_data(data)
    
    # ודא שהעמודות המתאימות נמצאות בנתונים המוכנים
    missing_cols = set(X.columns) - set(prepared_data.columns)
    for col in missing_cols:
        prepared_data[col] = 0
    
    prepared_data = prepared_data[X.columns]

    # חיזוי המחיר באמצעות המודל
    prediction = model.predict(prepared_data)[0]

    return render_template('index.html', prediction_text=f'Predicted Price: {prediction}')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
