# Temperature Prediction Models

This project includes Python scripts implementing different regression models to predict temperatures based on given features. The models covered in this project are:

- **Linear Regression**: Basic linear regression model for temperature prediction.
- **Polynomial Regression**: Extends linear regression by adding polynomial features to the input data.
- **Ridge Regression**: Applies L2 regularization to linear regression to prevent overfitting.
- **Lasso Regression**: Applies L1 regularization to linear regression to encourage sparsity in the coefficients.
- **Support Vector Regression (SVR)**: Utilizes support vector machines for regression tasks.

## Files

- **linear_regression.py**: Implements linear regression model for temperature prediction.
- **polynomial_regression.py**: Implements polynomial regression model for temperature prediction.
- **ridge_regression.py**: Implements ridge regression model for temperature prediction.
- **lasso_regression.py**: Implements lasso regression model for temperature prediction.
- **support_vector_regression.py**: Implements support vector regression model for temperature prediction.
- **main.py**: Entry point script that orchestrates the data processing, model training, evaluation, and prediction.
- **preprocess.py**: Preprocessing script that converts the original dataset into a suitable format for modeling.

## Preprocessing

The `preprocess.py` script is responsible for preprocessing the dataset before it is used for training the regression models. Here's what it does:

1. **Conversion to Float**: It converts numerical values represented as strings with comma as decimal separator to float format.
2. **Replacing Negative Values**: It replaces negative values with the median of the corresponding feature column.

## Usage

To use these models for temperature prediction, follow these steps:

1. Ensure you have Python installed along with necessary libraries (e.g., scikit-learn, pandas).
2. Run `preprocess.py` to preprocess the dataset and save the preprocessed data.
3. Modify `main.py` to load the preprocessed dataset and adjust any parameters if needed.
4. Run `main.py`. This will train each regression model, evaluate them, make predictions, and save the results.

## Example

```bash
python preprocess.py
python main.py
```

## Output

The output will include the Mean Squared Error (MSE) values for each regression model, along with the predictions saved in a CSV file named temperature_prediction.csv under the results directory.
