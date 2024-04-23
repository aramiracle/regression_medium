import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import Lasso, LassoCV

def predict_temperature_lasso(X_train, X_test, y_train, y_test, poly_degree=3, alpha_range=(-5, 5), num_alphas=100):
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly_features.fit_transform(X_train)  # Transform training features
    X_test_poly = poly_features.transform(X_test)       # Transform testing features using the same polynomial features

    # Generate a range of alpha values for Lasso regularization
    alphas = np.logspace(alpha_range[0], alpha_range[1], num=num_alphas)

    # Initialize and fit LassoCV model to select the best alpha value via cross-validation
    lasso_cv_model = LassoCV(alphas=alphas, max_iter=100000, tol=1e-2)
    lasso_cv_model.fit(X_train_poly, y_train)

    # Print the best alpha found during cross-validation
    print("Best Alpha for Ridge:", lasso_cv_model.alpha_)

    # Initialize Lasso model with the best alpha value selected by LassoCV, then fit it on the entire training set
    lasso_cv_model_best_alpha = Lasso(alpha=lasso_cv_model.alpha_, max_iter=100000, tol=1e-2)
    lasso_cv_model_best_alpha.fit(X_train_poly, y_train)

    # Predict the target variable using the trained Lasso model on the testing data
    y_pred_best_alpha = lasso_cv_model_best_alpha.predict(X_test_poly)

    # Calculate the mean squared error (MSE) between the actual and predicted values
    mse = mean_squared_error(y_test, y_pred_best_alpha)
    
    # Calculate R-squared
    mape = mean_absolute_percentage_error(y_test, y_pred_best_alpha)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred_best_alpha)

    # Return the predicted values, MSE, and R-squared
    return y_pred_best_alpha, (mse, mape, r2)
