# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 08:06:37 2025

@author: avina
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import Dataset, Reader, KNNWithMeans, SVD
from surprise.model_selection import GridSearchCV, train_test_split
from math import sqrt


df_ar = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Recommender systems/train-201102-092914.csv')
df_ai = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Recommender systems/Article_Recommendation/article_info.csv')
df_ss = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Recommender systems/Article_Recommendation/sample_submission.csv')
df_test = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Recommender systems/Article_Recommendation/test.csv')
df_ar.columns
df_ar.dtypes
df_ar.shape
df_ai.columns
df_ai.dtypes
df_ai.shape
df_ss.columns
df_ss.dtypes
df_ss.shape
df_test.columns
df_test.dtypes
df_test.shape

train_data = df_ar.merge(df_ai, on="article_id", how="left")
train_data.shape

reader = Reader(rating_scale=(1, 2))
data = Dataset.load_from_df(train_data[['user_id', 'article_id', 'rating']], reader)
# Train-test split
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

#Collaborative filtering

# Grid search for KNNWithMeans 
param_grid = {
    "k": list(range(1, 50, 5)),
    "sim_options": {
        "name": ["cosine", "pearson"],
        "user_based": [True]
    }
}

gs_user = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], cv=5, n_jobs=-1)
gs_user.fit(data)


best_user_model = gs_user.best_estimator['rmse']
best_user_model.fit(trainset)
pred_user = best_user_model.test(testset)
rmse_user = sqrt(mean_squared_error([pred.r_ui for pred in pred_user], [pred.est for pred in pred_user]))

# Best results
print("User-Based CF - Best RMSE:", rmse_user)
print("Best Parameters (User-Based):", gs_user.best_params['rmse'])

# Item-Based Collaborative Filtering

param_grid_item = {
    "k": list(range(1, 50, 5)),
    "sim_options": {"name": ["cosine", "pearson"], "user_based": [False]}
}
gs_item = GridSearchCV(KNNWithMeans, param_grid_item, measures=['rmse'], cv=5, n_jobs=-1)
gs_item.fit(data)
best_item_model = gs_item.best_estimator['rmse']
best_item_model.fit(trainset)
pred_item = best_item_model.test(testset)
rmse_item = sqrt(mean_squared_error([pred.r_ui for pred in pred_item], [pred.est for pred in pred_item]))
print("Item-Based CF - Best RMSE:", rmse_item)
print("Best Parameters (Item-Based):", gs_item.best_params['rmse'])

#Matrix Factorization (SVD)

param_grid_svd = {'n_factors':list(range(1,50,5)), 'n_epochs': [5, 10, 20], 'random_state': [42]}

gs_svd = GridSearchCV(SVD, 
                  param_grid_svd, 
                  measures=['rmse'], 
                  cv=5, 
                  n_jobs = -1)

gs_svd.fit(data)

best_svd_model = gs_svd.best_estimator['rmse']
best_svd_model.fit(trainset)
pred_svd = best_svd_model.test(testset)

rmse_svd = sqrt(mean_squared_error([pred.r_ui for pred in pred_svd], [pred.est for pred in pred_svd]))
print("SVD (Matrix Factorization) - Best RMSE:", rmse_svd)
print("Best Parameters (SVD):", gs_svd.best_params['rmse'])

# Make predictions on test set
def predict_ratings(model, test_data):
    return pd.DataFrame([
        {
            "user_id": row["user_id"],
            "article_id": row["article_id"],
            "rating": model.predict(row["user_id"], row["article_id"]).est
        }
        for _, row in test_data.iterrows()
    ])

pred_user_df = predict_ratings(best_user_model, df_test)
pred_item_df = predict_ratings(best_item_model, df_test)
pred_svd_df = predict_ratings(best_svd_model, df_test)

print("User-Based RMSE:", rmse_user)
print("Item-Based RMSE:", rmse_item)
print("SVD RMSE:", rmse_svd)

print(pred_user_df.head())
print(pred_item_df.head())
print(pred_svd_df.head())

submission_user = df_ss.copy()
submission_item = df_ss.copy()
submission_svd = df_ss.copy()

submission_user['rating'] = pred_user_df['rating']
submission_item['rating'] = pred_item_df['rating']
submission_svd['rating'] = pred_svd_df['rating']

submission_user.to_csv("submission_user_based.csv", index=False)
submission_item.to_csv("submission_item_based.csv", index=False)
submission_svd.to_csv("submission_svd.csv", index=False)
