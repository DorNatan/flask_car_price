import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from car_data_prep import prepare_data

# קריאת קובץ הנתונים 
datafile = "https://raw.githubusercontent.com/DorNatan/flask_car_price/main/dataset.csv"
dataset = pd.read_csv(datafile)

# הכנת הדאטה
prepared_data = prepare_data(dataset)

# הפרדת התכונות (X) והיעד (y)
X = prepared_data.drop(columns=['Price'])
y = prepared_data['Price']

# אימון המודל ElasticNet עם הפרמטרים הטובים ביותר שנמצאו קודם
best_model = ElasticNet(alpha=0.01, l1_ratio=0.9)
best_model.fit(X, y)

# ביצוע קרוס-וולידציה עם 10 קפלים
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

# הדפסת התוצאות
print("Cross-Validated RMSE Scores:", cv_rmse_scores)
print("Mean Cross-Validated RMSE:", cv_rmse_scores.mean())

