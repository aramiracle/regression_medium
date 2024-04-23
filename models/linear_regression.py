from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def predict_temperature_linear(X_train, X_test, y_train, y_test):
    # Step 3: Choose a regression model
    model = LinearRegression()  # Example: Linear Regression

    # Step 4: Train the model
    model.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = model.predict(X_test)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Step 6: Return predictions
    return y_pred, mse