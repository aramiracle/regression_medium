import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing functions from custom regression models
from models.linear_regression import predict_temperature_linear
from models.polynomial_regression import predict_temperature_polynomial
from models.ridge_regression import predict_temperature_ridge
from models.lasso_regression import predict_temperature_lasso
from models.support_vector_regression import predict_temperature_svr

if __name__ == '__main__':
    # Step 1: Load the dataset
    df = pd.read_csv('dataset/air_quality_converted.csv')

    predicted_column_name = 'T'  # Name of the target column (temperature)
    
    # Step 2: Split data into features (X) and target (y)
    X = df.drop(columns=[predicted_column_name])  # Features
    y = df[predicted_column_name]  # Target

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Make predictions using different regression models
    predictions = {
        'Linear' : predict_temperature_linear(X_train, X_test, y_train, y_test),
        'Polynomial' : predict_temperature_polynomial(X_train, X_test, y_train, y_test),
        'Ridge' : predict_temperature_ridge(X_train, X_test, y_train, y_test),
        'Lasso' : predict_temperature_lasso(X_train, X_test, y_train, y_test),
        'SVR' : predict_temperature_svr(X_train, X_test, y_train, y_test)
    }

    # Step 5: Store Mean Squared Error (MSE) values in a dictionary
    mse_values = {}
    for model_name, prediction_data in predictions.items():
        mse_values[model_name] = prediction_data[1]

    # Step 6: Print MSE values for each regression model
    print("MSE Values:")
    for model_name, mse in mse_values.items():
        print(f"{model_name} Regression: {round(mse, 5)}")   

    # Step 7: Save predictions to a CSV file
    predicted_df = pd.DataFrame({
        'Real Values' : y_test,
        'Predicted Linear' : predictions['Linear'][0],
        'Predicted Polynomial' : predictions['Polynomial'][0],
        'Predicted Ridge' : predictions['Ridge'][0],
        'Predicted Lasso' : predictions['Lasso'][0],
        'Predicted SVR' : predictions['SVR'][0]
    }).reset_index(drop=True)

    os.makedirs('results', exist_ok=True)  # Create a directory to store results if it doesn't exist

    # Save the DataFrame to a CSV file
    predicted_df.to_csv('results/temperature_prediction.csv')
