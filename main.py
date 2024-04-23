import os
import pandas as pd
from sklearn.model_selection import train_test_split

from models.linear_regression import predict_temperature_linear
from models.polynomial_regression import predict_temperature_polynomial
from models.ridge_regression import predict_temperature_ridge
from models.lasso_regression import predict_temperature_lasso
from models.support_vector_regression import predict_temperature_svr

if __name__ == '__main__':
    df = pd.read_csv('dataset/air_quality_converted.csv')

    predicted_column_name = 'T'
    # Step 1: Split data into features (X) and target (y)
    X = df.drop(columns=[predicted_column_name])  # Features
    y = df[predicted_column_name]  # Target

    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictions = {
        'Linear' : predict_temperature_linear(X_train, X_test, y_train, y_test),
        'Polynomial' : predict_temperature_polynomial(X_train, X_test, y_train, y_test),
        'Ridge' : predict_temperature_ridge(X_train, X_test, y_train, y_test),
        'Lasso' : predict_temperature_lasso(X_train, X_test, y_train, y_test),
        'SVR' : predict_temperature_svr(X_train, X_test, y_train, y_test)
    }

    # Store MSE values in a dictionary
    mse_values = {}
    for model_name, prediction_data in predictions.items():
        mse_values[model_name] = prediction_data[1]

    # Print MSE values
    print("MSE Values:")
    for model_name, mse in mse_values.items():
        print(f"{model_name} Regression: {round(mse, 5)}")   

    # Save predictions to CSV
    predicted_df = pd.DataFrame({
        'Real Values' : y_test,
        'Predicted Linear' : predictions['Linear'][0],
        'Predicted Polynomial' : predictions['Polynomial'][0],
        'Predicted Ridge' : predictions['Ridge'][0],
        'Predicted Lasso' : predictions['Lasso'][0],
        'Predicted SVR' : predictions['SVR'][0]
    }).reset_index(drop=True)

    os.makedirs('results', exist_ok=True)

    predicted_df.to_csv('results/temperature_prediction.csv')