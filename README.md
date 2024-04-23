# Temperature Prediction Models

This project includes Python scripts implementing different regression models to predict temperatures based on given features. The models covered in this project are:

- **Linear Regression**: Basic linear regression model for temperature prediction.
- **Polynomial Regression**: Extends linear regression by adding polynomial features to the input data.
- **Ridge Regression**: Applies L2 regularization to linear regression to prevent overfitting.
- **Lasso Regression**: Applies L1 regularization to linear regression to encourage sparsity in the coefficients.
- **Support Vector Regression (SVR)**: Utilizes support vector machines for regression tasks.

## Dataset
The dataset used for this project is the [Air Quality Data Set](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set) available on Kaggle. It provides various features including temperature, humidity, and air quality measurements.

## Files

- **models/linear_regression.py**: Implements linear regression model for temperature prediction.
- **models/polynomial_regression.py**: Implements polynomial regression model for temperature prediction.
- **models/ridge_regression.py**: Implements ridge regression model for temperature prediction.
- **models/lasso_regression.py**: Implements lasso regression model for temperature prediction.
- **models/support_vector_regression.py**: Implements support vector regression model for temperature prediction.
- **main.py**: Entry point script that orchestrates the data processing, model training, evaluation, and prediction.
- **preprocess.py**: Preprocessing script that converts the original dataset into a suitable format for modeling.

## Preprocessing

The `preprocess.py` script is responsible for preprocessing the dataset before it is used for training the regression models. Here's what it does:

1. **Conversion to Float**: It converts numerical values represented as strings with comma as decimal separator to float format.
2. **Replacing Negative Values**: It replaces negative values with the median of the corresponding feature column.

## Steps for prediction in `main.py`

1. **Load the dataset**: Load the dataset containing air quality features and temperature values.

2. **Split Data**: Split the dataset into features (X) and target (y), where X represents the input features and y represents the target variable (temperature).

3. **Split Data into Training and Testing Sets**: Divide the dataset into training and testing sets to train the model on one set and evaluate its performance on another.

4. **Normalize Data**: Standardize the features by scaling them to have a mean of 0 and a standard deviation of 1.

5. **Make Predictions**: Use different regression models to make temperature predictions. The models used in this project include:
   - Linear Regression
   - Polynomial Regression
   - Ridge Regression
   - Lasso Regression
   - Support Vector Regression (SVR)

6. **Calculate Mean Squared Error (MSE)**: Compute the Mean Squared Error (MSE) for each regression model to evaluate its performance.

7. **Print MSE Values**: Display the MSE values for each regression model.

8. **Save Predictions to CSV File**: Store the predicted temperature values along with the real temperature values in a CSV file for further analysis.

## Usage

To use these models for temperature prediction, follow these steps:

1. Ensure you have Python installed along with necessary libraries (scikit-learn, pandas, NumPy). You can install them like below in command line:

```
pip install -r requirements.txt
```

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
