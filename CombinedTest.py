import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Function to generate synthetic time series data
def generate_synthetic_data(start_date, end_date, freq='D', trend_slope=0.5, amplitude=10, noise_std=5):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    time_values = np.arange(len(date_range))

    linear_trend = trend_slope * time_values
    seasonal_pattern = amplitude * np.sin(2 * np.pi * time_values / 365)
    noise = np.random.normal(0, noise_std, len(date_range))

    synthetic_data = linear_trend + seasonal_pattern + noise
    synthetic_df = pd.DataFrame({'value': synthetic_data}, index=date_range)

    return synthetic_df.sort_index()

# Function to preprocess data and create input sequences for LSTM
def prepare_data(series, look_back=10):
    data = series.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:(i + look_back), 0])
        y.append(data_scaled[i + look_back, 0])

    return np.array(X), np.array(y), scaler

# Function to train the LSTM model over multiple runs and test the classifier
# Function to train the LSTM model over multiple runs and test the classifier
def train_and_test_classifier(num_runs=100, look_back=5, epochs=100, lstm_units=50, rf_estimators=100):
    upper_bound_accuracy = []
    lower_bound_accuracy = []

    for run in range(num_runs):
        # Generate synthetic data for each run
        synthetic_data = generate_synthetic_data('2022-01-01', '2023-01-01')

        # Prepare data for LSTM
        X, y, scaler = prepare_data(synthetic_data, look_back)

        # Reshape data for LSTM input
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Adjust the reshaping here

        # Create LSTM model
        model = Sequential()
        model.add(LSTM(units=lstm_units, input_shape=(look_back, 1), activation='relu'))  # Adjust the input shape here
        model.add(Dense(units=1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the LSTM model
        model.fit(X, y, epochs=epochs, batch_size=1, verbose=0)

        # Train Random Forest regressor for upper and lower bounds
        X_train, X_test, y_train, y_test = train_test_split(X.reshape(X.shape[0], -1),
                                                            synthetic_data['value'][look_back:], test_size=0.2,
                                                            shuffle=False)

        upper_bound_regressor = RandomForestRegressor(n_estimators=rf_estimators, random_state=42)
        lower_bound_regressor = RandomForestRegressor(n_estimators=rf_estimators, random_state=42)

        upper_bound_regressor.fit(X_train, y_train)
        lower_bound_regressor.fit(X_train, y_train)

        # Make predictions on the test data
        upper_bound_predictions = upper_bound_regressor.predict(X_test)
        lower_bound_predictions = lower_bound_regressor.predict(X_test)

        # Evaluate the accuracy of the classifiers
        upper_bound_accuracy.append(accuracy_score(y_test > upper_bound_predictions, y_test > scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()))
        lower_bound_accuracy.append(accuracy_score(y_test < lower_bound_predictions, y_test < scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()))

    # Calculate and print the average accuracy over all runs
    avg_upper_bound_accuracy = np.mean(upper_bound_accuracy)
    avg_lower_bound_accuracy = np.mean(lower_bound_accuracy)

    print(f'Average Upper Bound Accuracy: {avg_upper_bound_accuracy}')
    print(f'Average Lower Bound Accuracy: {avg_lower_bound_accuracy}')

# Call the training and testing function
train_and_test_classifier(num_runs=100, look_back=5, epochs=100, lstm_units=50, rf_estimators=100)