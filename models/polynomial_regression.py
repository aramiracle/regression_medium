from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def predict_temperature_polynomial(X_train, X_test, y_train, y_test, poly_degree=3):
    # Step 5: Choose a regression model
    # Example: Polynomial Regression
    poly_features = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()

    # Step 4: Train the model
    model.fit(X_train_poly, y_train)

    # Step 5: Evaluate the model
    y_pred = model.predict(X_test_poly)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    return y_pred, mse