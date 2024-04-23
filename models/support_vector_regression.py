from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

def predict_temperature_svr(X_train, X_test, y_train, y_test):
    # Step 1: Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 2: Choose a regression model
    # Example: Support Vector Regression (SVR)
    svr_model = SVR(kernel='rbf')  # Radial Basis Function (RBF) kernel

    # Step 3: Train the model
    svr_model.fit(X_train_scaled, y_train)

    # Step 4: Evaluate the model
    y_pred = svr_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    return y_pred, mse