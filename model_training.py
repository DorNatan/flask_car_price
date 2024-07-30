import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import pickle
from car_data_prep import prepare_data  # ודא שהפונקציה הזו קיימת בקובץ car_data_prep.py

# קריאת קובץ הנתונים 
datafile = "https://raw.githubusercontent.com/DorNatan/flask_car_price/main/dataset.csv"
dataset = pd.read_csv(datafile)

# הכנת הדאטה
prepared_data = prepare_data(dataset)

# הפרדת התכונות (X) והיעד (y)
X = prepared_data.drop(columns=['Price'])
y = prepared_data['Price']

# חיפוש פרמטרים אופטימליים באמצעות GridSearchCV
param_grid = {
    'alpha': [0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.5, 0.9]
}

grid_search = GridSearchCV(ElasticNet(), param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# הדפסת הפרמטרים הטובים ביותר שנמצאו
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# אימון המודל ElasticNet עם הפרמטרים הטובים ביותר שנמצאו
best_model = ElasticNet(**best_params)
best_model.fit(X, y)

# ביצוע קרוס-וולידציה עם 10 קפלים
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

# הדפסת התוצאות
print("Cross-Validated RMSE Scores:", cv_rmse_scores)
print("Mean Cross-Validated RMSE:", cv_rmse_scores.mean())

# שמירת המודל המאומן בקובץ PKL
with open("trained_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)
