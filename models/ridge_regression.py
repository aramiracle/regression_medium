import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV

def predict_temperature_ridge(X_train, X_test, y_train, y_test, poly_degree=3, alpha_range=(-5, 5), num_alphas=100):
    # Step 5: Choose a regression model
    # Example: Polynomial Regression with RidgeCV regularization
    poly_features = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Define alpha values ranging from 1e-3 to 1e5
    alphas = np.logspace(alpha_range[0], alpha_range[1], num=num_alphas)

    # Use RidgeCV for automatic alpha selection
    ridge_cv_model = RidgeCV(alphas=alphas)
    ridge_cv_model.fit(X_train_poly, y_train)

    # Print the selected alpha
    print("Best Alpha for Ridge:", ridge_cv_model.alpha_)

    # Refit the model with the best alpha
    ridge_cv_model_best_alpha = Ridge(alpha=ridge_cv_model.alpha_)
    ridge_cv_model_best_alpha.fit(X_train_poly, y_train)

    # Evaluate the model with the best alpha
    y_pred_best_alpha = ridge_cv_model_best_alpha.predict(X_test_poly)
    mse_best_alpha = mean_squared_error(y_test, y_pred_best_alpha)

    return y_pred_best_alpha, mse_best_alpha