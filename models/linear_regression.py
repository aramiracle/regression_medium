from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def predict_temperature_linear(X_train, X_test, y_train, y_test):
    # Create a Linear Regression model
    model = LinearRegression()
    
    # Train the model using the training data
    model.fit(X_train, y_train)
    
    # Use the trained model to make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate the mean squared error (MSE) between the actual and predicted values
    mse = mean_squared_error(y_test, y_pred)

    # Return the predicted values and the mean squared error
    return y_pred, mse