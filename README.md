# Algerian Forest Fire Prediction using Ridge Regression

This project focuses on building a machine learning regression model to predict the Fire Weather Index (FWI) using meteorological and fire-related features from the Algerian Forest Fires dataset.
The trained model is later serialized and deployed using a Flask web application for real-time prediction.

Problem Statement

Forest fires are strongly influenced by environmental conditions such as temperature, humidity, wind speed, and moisture indices.
The goal of this project is to predict the FWI (Fire Weather Index), which represents the intensity and spread potential of fires, using historical weather and fire data.

This is formulated as a supervised regression problem.

Dataset Description

The dataset contains 243 records with meteorological, fire index, and regional information collected from two regions in Algeria.

After preprocessing, the main features used include temperature, relative humidity, wind speed, rainfall, FFMC, DMC, DC, ISI, BUI, fire class, and region.

The target variable is FWI (Fire Weather Index).

Data Cleaning and Preprocessing

Initial inspection showed redundant date-related columns (day, month, year), which were removed as they did not add predictive value.

The Classes column contained inconsistent string labels such as "fire" and "not fire" with spacing issues.
This column was cleaned and converted into a binary numerical format where:

1 represents fire

0 represents no fire

Correlation analysis was performed using a heatmap.
Highly correlated features (correlation > 0.85) were identified, and the feature DC was removed to reduce multicollinearity.

All numerical features were standardized using StandardScaler to ensure stable model training.

Model Training

Multiple regression models were evaluated:

Linear Regression

Ridge Regression

Lasso Regression

ElasticNet Regression

The dataset was split into training and testing sets using a 75–25 split.

Ridge Regression performed exceptionally well, achieving an R² score of ~0.99 on the test set, indicating strong predictive capability while controlling overfitting.

ElasticNet performed reasonably well but showed reduced accuracy compared to Ridge.

Model Evaluation

Model performance was evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Repeated K-Fold Cross Validation (5 folds × 3 repeats) was applied to ensure robustness.

Ridge Regression consistently outperformed Linear and Lasso Regression across folds, confirming its suitability for deployment.

A baseline MAE was also computed using the mean FWI value, showing that the trained model significantly outperforms a naive baseline.

Model Serialization

After training and validation, the following artifacts were saved using pickle:

ridge.pkl → trained Ridge Regression model

scaler.pkl → fitted StandardScaler

These files are later loaded by the Flask application for inference.

Project Structure:

Forest-Fire-Prediction/
│
├── notebooks/
│   └── Algerian_forest_fire_model_training.ipynb
│
├── models/
│   ├── ridge.pkl
│   └── scaler.pkl
│
├── templates/
│   └── index1.html
│
├── application.py
├── requirements.txt
└── README.md

Deployment Overview:

The trained model is deployed using Flask.
User inputs are collected via an HTML form, scaled using the saved StandardScaler, and passed to the Ridge model for prediction.

The predicted FWI value is rendered dynamically on the web interface.

Key Learnings:

This project demonstrates a complete end-to-end machine learning workflow, including data cleaning, feature engineering, correlation analysis, regression modeling, cross validation, serialization, and deployment.

It highlights the importance of feature scaling, multicollinearity handling, and proper model evaluation before deployment.

Future Improvements

The project can be extended by adding hyperparameter tuning, confidence intervals, real-time weather data integration, and cloud deployment using AWS or Docker.

Author

Bedant Behera
GitHub: https://github.com/Bedant03
