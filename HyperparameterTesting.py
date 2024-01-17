import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns

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

# Function to split data into train and test sets using time-based cross-validation
def timeseries_train_test_split(data, test_size):
    split_idx = int(len(data) * (1 - test_size))
    train, test = data[:split_idx], data[split_idx:]
    return train, test

# Function to prepare data for LSTM
def prepare_data_for_lstm(data, look_back=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    x, y = [], []
    for i in range(len(scaled_data) - look_back):
        x.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])

    return np.array(x), np.array(y), scaler

# Function to create and evaluate LSTM model
def evaluate_lstm_model(train, test, look_back, units, epochs):
    x_train, y_train, scaler = prepare_data_for_lstm(train, look_back)
    x_test, y_test, scaler2 = prepare_data_for_lstm(test, look_back)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=units, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model multiple times
    mse_values = []
    for _ in range(10):  # Adjust the number of runs as needed
        model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=0)
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        test_values = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = mean_squared_error(test_values, predictions)
        mse_values.append(mse)

    return np.mean(mse_values), mse_values

# Function to perform time series cross-validation and test different LSTM hyperparameters
def test_lstm_hyperparameters(data, test_size):
    train, test = timeseries_train_test_split(data, test_size)

    hyperparameters = {
        'look_back_values': [1, 2, 3],
        'units_values': [50, 200, 300],
        'epochs_values': [50, 100, 300]
    }

    results = {}

    for look_back in hyperparameters['look_back_values']:
        for units in hyperparameters['units_values']:
            for epochs in hyperparameters['epochs_values']:
                avg_mse, individual_mse = evaluate_lstm_model(train, test, look_back, units, epochs)
                results[(look_back, units, epochs)] = (avg_mse, individual_mse)

    return results

# Plotting function
def plot_lstm_results(results):
    df_results = pd.DataFrame(list(results.values()),
                              index=pd.MultiIndex.from_tuples(results.keys(), names=['Look Back', 'Units', 'Epochs']),
                              columns=['Avg MSE', 'Individual MSE'])

    plt.figure(figsize=(15, 8))

    # Plot for different look_backs
    plt.subplot(3, 1, 1)
    sns.lineplot(data=df_results, x='Look Back', y='Avg MSE', hue='Epochs', style='Units', markers=True, dashes=False)
    plt.title('Avg MSE vs Look Back')
    plt.xlabel('Look Back')
    plt.ylabel('Mean Squared Error')
    plt.legend(title='Epochs')

    # Plot for different units
    plt.subplot(3, 1, 2)
    sns.lineplot(data=df_results, x='Units', y='Avg MSE', hue='Epochs', style='Look Back', markers=True, dashes=False)
    plt.title('Avg MSE vs Units')
    plt.xlabel('Units')
    plt.ylabel('Mean Squared Error')
    plt.legend(title='Epochs')

    # Plot for different epochs
    plt.subplot(3, 1, 3)
    sns.lineplot(data=df_results, x='Epochs', y='Avg MSE', hue='Look Back', style='Units', markers=True, dashes=False)
    plt.title('Avg MSE vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend(title='Look Back')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate synthetic time series data
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    synthetic_data = generate_synthetic_data(start_date, end_date)

    # Set the test_size parameter for time series cross-validation
    test_size = 0.2

    # Test multiple LSTM hyperparameters
    results = test_lstm_hyperparameters(synthetic_data['value'], test_size)

    # Plot the results
    plot_lstm_results(results)
    """
if __name__ == "__main__":
    # Generate synthetic time series data
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    synthetic_data = generate_synthetic_data(start_date, end_date)

    # Set the test_size parameter for time series cross-validation
    test_size = 0.2

    # Best hyperparameters
    best_look_back = 7
    best_units = 15
    best_epochs = 15

    # Train the LSTM model over 1000 runs
    mse_values = []
    for run in range(1, 1001):
        # Split data into training and testing sets
        train, test = timeseries_train_test_split(synthetic_data['value'], test_size)

        # Evaluate the LSTM model with the best hyperparameters
        avg_mse, _ = evaluate_lstm_model(train, test, best_look_back, best_units, best_epochs)
        mse_values.append(avg_mse)

        # Print the timestamp and average mean squared error for each run
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Run {run} completed at {timestamp}. Average Mean Squared Error: {avg_mse}")

    # Print the overall average mean squared error over 1000 runs
    overall_avg_mse = np.mean(mse_values)
    print(f"\nOverall Average Mean Squared Error over 1000 runs: {overall_avg_mse}")
    
    """