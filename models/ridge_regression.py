import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV

def predict_temperature_ridge(X_train, X_test, y_train, y_test, poly_degree=3, alpha_range=(-5, 5), num_alphas=100):
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly_features.fit_transform(X_train)  # Transform training data
    X_test_poly = poly_features.transform(X_test)  # Transform test data using the same polynomial features

    # Generate a range of alpha values
    alphas = np.logspace(alpha_range[0], alpha_range[1], num=num_alphas)

    # Perform Ridge regression with cross-validation to find the best alpha
    ridge_cv_model = RidgeCV(alphas=alphas)
    ridge_cv_model.fit(X_train_poly, y_train)

    # Print the best alpha found during cross-validation
    print("Best Alpha for Ridge:", ridge_cv_model.alpha_)

    # Create a Ridge regression model with the best alpha
    ridge_cv_model_best_alpha = Ridge(alpha=ridge_cv_model.alpha_)
    ridge_cv_model_best_alpha.fit(X_train_poly, y_train)

    # Predict the target variable using the model with the best alpha
    y_pred_best_alpha = ridge_cv_model_best_alpha.predict(X_test_poly)

    # Calculate mean squared error between predicted and actual values
    mse_best_alpha = mean_squared_error(y_test, y_pred_best_alpha)

    # Return predicted values and mean squared error
    return y_pred_best_alpha, mse_best_alpha
