from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

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
    
    # Calculate mean squared error between predicted and actual values
    mse = mean_squared_error(y_test, y_pred)
    
    # Return predicted values and mean squared error
    return y_pred, mse
