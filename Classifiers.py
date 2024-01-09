import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results):
    df_results = pd.DataFrame(list(results.values()), index=pd.MultiIndex.from_tuples(results.keys(), names=['Look Back', 'n_estimators', 'max_depth']), columns=['MSE'])
    print(df_results)
    plt.figure(figsize=(15, 8))
    ax = sns.FacetGrid(df_results.reset_index(), col='n_estimators', row='max_depth', hue='n_estimators', palette='viridis')
    ax.map(sns.scatterplot, 'Look Back', 'MSE', alpha=0.7, s=100)
    ax.set_axis_labels('Look Back', 'Mean Squared Error')
    ax.add_legend()

    plt.tight_layout()
    plt.show()

# Function to test models, record results, and plot using Seaborn
def test_models_and_plot(data, test_size, look_back_values, n_estimators_values, max_depth_values):
    train, test = timeseries_train_test_split(data, test_size)

    results = {}

    for look_back in look_back_values:
        X_train_rf, y_train_rf = prepare_data_for_random_forest(train, look_back)
        X_test_rf, y_test_rf = prepare_data_for_random_forest(test, look_back)

        for n_estimators in n_estimators_values:
            for max_depth in max_depth_values:
                print(f"\nTesting with Look Back = {look_back}, n_estimators = {n_estimators}, max_depth = {max_depth}")

                # Random Forest
                mse, _ = evaluate_random_forest_model(X_train_rf, X_test_rf, y_train_rf, y_test_rf, n_estimators, max_depth)
                results[(look_back, n_estimators, max_depth)] = mse

    # Plot the results using Seaborn
    plot_results(results)


# Modified function to generate mock time series data with events
def generate_mock_time_series_with_events(start_date, end_date, freq='D', trend_slope=0.5, amplitude=10, noise_std=5, event_probability=0.05):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    time_values = np.arange(len(date_range))

    linear_trend = trend_slope * time_values
    seasonal_pattern = amplitude * np.sin(2 * np.pi * time_values / 365)
    noise = np.random.normal(0, noise_std, len(date_range))

    synthetic_data = linear_trend + seasonal_pattern + noise
    synthetic_df = pd.DataFrame({'value': synthetic_data}, index=date_range)

    # Adding a binary target variable (A or B)
    synthetic_df['target'] = np.where(synthetic_df['value'] > synthetic_df['value'].shift(1), 'B', 'A')

    # Adding events as binary indicators
    synthetic_df['positive_event'] = np.random.choice([0, 1], size=len(synthetic_df), p=[1 - event_probability, event_probability])
    synthetic_df['negative_event'] = np.random.choice([0, 1], size=len(synthetic_df), p=[1 - event_probability, event_probability])

    # Adjusting target variable based on events
    synthetic_df.loc[synthetic_df['positive_event'] == 1, 'target'] = 'B'
    synthetic_df.loc[synthetic_df['negative_event'] == 1, 'target'] = 'A'

    return synthetic_df

# Function to split data into train and test sets using time-based cross-validation
def timeseries_train_test_split(data, test_size):
    split_idx = int(len(data) * (1 - test_size))
    train, test = data[:split_idx], data[split_idx:]
    return train, test

# Function to prepare data for logistic regression
def prepare_data_for_logistic_regression(data):
    X = data[['value', 'positive_event', 'negative_event']].values
    y = data['target'].values
    return X, y

# Function to prepare data for random forest
def prepare_data_for_random_forest(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[['value', 'positive_event', 'negative_event']].values[i:(i + look_back)].flatten())
        y.append(data['target'].values[i + look_back])

    return np.array(X), np.array(y)

# Function to create and evaluate Logistic Regression model
def evaluate_logistic_regression_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Logistic Regression:")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

# Function to create and evaluate Random Forest model
def evaluate_random_forest_model(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Convert predictions to numeric type
    y_pred_numeric = np.where(y_pred == 'B', 1, 0).astype(float)

    # Convert true labels to numeric type
    y_test_numeric = np.where(y_test == 'B', 1, 0).astype(float)

    # Evaluation
    mse = ((y_pred_numeric - y_test_numeric) ** 2).mean()  # Mean Squared Error
    report = classification_report(y_test, y_pred)

    print("\nRandom Forest:")
    print(f"Mean Squared Error: {mse}")
    print(f"Classification Report:\n{report}")

    return mse, y_pred  # Return MSE and predictions

# Function to perform time series cross-validation and test different models
def test_models(data, test_size, look_back_values, n_estimators_values, max_depth_values):
    train, test = timeseries_train_test_split(data, test_size)

    for look_back in look_back_values:
        X_train_rf, y_train_rf = prepare_data_for_random_forest(train, look_back)
        X_test_rf, y_test_rf = prepare_data_for_random_forest(test, look_back)

        for n_estimators in n_estimators_values:
            for max_depth in max_depth_values:
                print(f"\nTesting with Look Back = {look_back}, n_estimators = {n_estimators}, max_depth = {max_depth}")

                # Logistic Regression
                X_train_lr, y_train_lr = prepare_data_for_logistic_regression(train)
                X_test_lr, y_test_lr = prepare_data_for_logistic_regression(test)
                evaluate_logistic_regression_model(X_train_lr, X_test_lr, y_train_lr, y_test_lr)

                # Random Forest
                evaluate_random_forest_model(X_train_rf, X_test_rf, y_train_rf, y_test_rf, n_estimators, max_depth)

# Generate mock time series data with events
start_date = '2022-01-01'
end_date = '2022-12-31'
mock_data = generate_mock_time_series_with_events(start_date, end_date, event_probability=0.1)

# Set the test_size parameter for time series cross-validation
test_size = 0.2

# Define hyperparameter values to test
look_back_values = [3, 5, 7]
n_estimators_values = [50, 100, 150]
max_depth_values = [None, 5, 10]

# Test models
test_models_and_plot(mock_data, test_size, look_back_values, n_estimators_values, max_depth_values)
