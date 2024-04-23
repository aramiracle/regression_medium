from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def predict_temperature_polynomial(X_train, X_test, y_train, y_test, poly_degree=3):
    # Create PolynomialFeatures object with specified degree
    poly_features = PolynomialFeatures(degree=poly_degree)
    
    # Transform training features to polynomial features
    X_train_poly = poly_features.fit_transform(X_train)
    
    # Transform test features to polynomial features using the same polynomial transformation
    X_test_poly = poly_features.transform(X_test)

    # Initialize Linear Regression model
    model = LinearRegression()

    # Fit the model on the polynomial features
    model.fit(X_train_poly, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test_poly)
    
    # Calculate the mean squared error (MSE) between the actual and predicted values
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate R-squared
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)

    # Return the predicted values, MSE, and R-squared
    return y_pred, (mse, mape, r2)
