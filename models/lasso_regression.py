import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV

def predict_temperature_lasso(X_train, X_test, y_train, y_test, poly_degree=3, alpha_range=(-5, 5), num_alphas=100):
    # Step 1: Feature engineering (if needed)
    # Example: Polynomial Features
    poly_features = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Define alpha values ranging from 1e-3 to 1e5
    alphas = np.logspace(alpha_range[0], alpha_range[1], num=num_alphas)

    # Use LassoCV for automatic alpha selection
    lasso_cv_model = LassoCV(alphas=alphas, max_iter=100000, tol=1e-1)
    lasso_cv_model.fit(X_train_poly, y_train)

    # Print the selected alpha
    print("Best Alpha for Lasso:", lasso_cv_model.alpha_)

    # Refit the model with the best alpha
    lasso_cv_model_best_alpha = Lasso(alpha=lasso_cv_model.alpha_, max_iter=100000, tol=1e-1)
    lasso_cv_model_best_alpha.fit(X_train_poly, y_train)

    # Evaluate the model with the best alpha
    y_pred_best_alpha = lasso_cv_model_best_alpha.predict(X_test_poly)
    mse_best_alpha = mean_squared_error(y_test, y_pred_best_alpha)

    return y_pred_best_alpha, mse_best_alpha
