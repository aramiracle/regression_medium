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

### 1. Load the Dataset
   - Load the dataset containing air quality data, particularly the temperature, from a CSV file.

### 2. Split Data into Features and Target
   - Separate the dataset into feature variables (X) and the target variable (y), where the target is the temperature.

### 3. Split Data into Training and Testing Sets
   - Split the dataset into training and testing sets using a ratio of 80:20, respectively, for model evaluation.

### 4. Normalize Data
   - Standardize the feature variables using `StandardScaler` to ensure that each feature contributes equally to the predictions.

### 5. Make Predictions Using Different Regression Models
   - Utilize various regression models to predict temperature:
     - Linear Regression
     - Polynomial Regression
     - Ridge Regression
     - Lasso Regression
     - Support Vector Regression (SVR)

### 6. Store Evaluation Values in a Dictionary
   - Store evaluation metrics, including Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), and R-squared (R2), for each regression model in a dictionary.

### 7. Print Evaluation Scores for Each Regression Model
   - Display the evaluation scores for each regression model, including MSE, MAPE, and R2.

### 8. Save Predictions to a CSV File
   - Create a DataFrame containing real temperature values and predicted values from each regression model.
   - Save the DataFrame to a CSV file named `temperature_prediction.csv` in the `results` directory.


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

Finally there is a medium article you can read for deeper insight. This is a [link](https://medium.com/@a.r.amouzad.m/classic-machine-learning-part-2-4-regression-24086d7cc374) to story.