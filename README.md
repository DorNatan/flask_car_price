# Car Price Prediction Web App

This repository contains a simple Flask application for predicting car prices using a trained machine learning model. The main components are:

- `app.py` – Flask web server exposing a form for user input and returning the predicted price.
- `data_preparation.py` – Functions for cleaning and preprocessing the car dataset.
- `train_model.py` – Script to train the ElasticNet model on the prepared data.
- `car_prices_dataset.csv` – Sample dataset used for training and feature extraction.
- `index.html` – Basic HTML template for the app's user interface.
- `trained_model.pkl` – Pretrained model used by the application.
- `requirements.txt` – Python dependencies.
- `TEAM_MEMBERS.txt` – List of project contributors.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (optional – a trained model is already provided):
   ```bash
   python train_model.py
   ```
3. Run the web application:
   ```bash
   python app.py
   ```
4. Open your browser at `http://localhost:5000` to use the app.
