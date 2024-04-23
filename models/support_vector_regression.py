from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

def predict_temperature_svr(X_train, X_test, y_train, y_test):
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler to training data and transform training data
    X_test_scaled = scaler.transform(X_test)       # Transform test data using the same scaler

    # Initialize SVR model with radial basis function (RBF) kernel
    svr_model = SVR(kernel='rbf')

    # Train SVR model on the scaled training data
    svr_model.fit(X_train_scaled, y_train)

    # Make predictions on the scaled test data
    y_pred = svr_model.predict(X_test_scaled)

    # Calculate mean squared error between predicted and true target values
    mse = mean_squared_error(y_test, y_pred)

    # Return predicted values and mean squared error
    return y_pred, mse
