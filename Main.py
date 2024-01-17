import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
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

    return np.mean(mse_values), mse_values, model


# Function to integrate LSTM and Random Forest
def integrate_lstm_random_forest(data, test_size, lstm_look_back, lstm_units, lstm_epochs, rf_n_estimators=100, rf_max_depth=None):
    # Generate LSTM predictions
    train, test = timeseries_train_test_split(data, test_size)
    print(train)
    print(test)
    avg_mse, _, lstm_model = evaluate_lstm_model(train, test, lstm_look_back, lstm_units, lstm_epochs)

    # Prepare data for Random Forest
    x_train_lstm, y_train_lstm, lstm_scaler = prepare_data_for_lstm(train, lstm_look_back)
    x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))
    lstm_predictions_train = lstm_model.predict(x_train_lstm)
    lstm_predictions_train = lstm_scaler.inverse_transform(lstm_predictions_train)

    # Prepare data for Random Forest
    x_test_lstm, y_test_lstm, lstm_scaler = prepare_data_for_lstm(test, lstm_look_back)
    x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0], x_test_lstm.shape[1], 1))
    lstm_predictions_test = lstm_model.predict(x_test_lstm)
    lstm_predictions_test = lstm_scaler.inverse_transform(lstm_predictions_test)


    # Ensure alignment of training and testing sets
    x_train_rf, y_train_rf = lstm_predictions_train[:-1], train['value'].values[lstm_look_back+1:]
    x_test_rf, y_test_rf = lstm_predictions_test[:-1], test['value'].values[lstm_look_back+1:]

    # Convert labels to numeric type
    y_train_rf_numeric = np.where(y_train_rf > 10, 2, np.where(y_train_rf > 5, 1, 0)).astype(float)
    y_test_rf_numeric = np.where(y_test_rf > 10, 2, np.where(y_test_rf > 5, 1, 0)).astype(float)

    # Train Random Forest Classifier
    model_rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
    model_rf.fit(x_train_rf, y_train_rf_numeric)

    # Predictions
    y_pred_rf = model_rf.predict(x_test_rf)

    # Convert predictions to original labels
    y_pred_rf_labels = np.where(y_pred_rf == 2, 'Unsafe', np.where(y_pred_rf == 1, 'Insignificant', 'Safe'))

    # Evaluation
    mse_rf = ((y_pred_rf - y_test_rf_numeric) ** 2).mean()  # Mean Squared Error
    report_rf = classification_report(y_test_rf_numeric, y_pred_rf)

    print("\nRandom Forest:")
    print(f"Mean Squared Error: {mse_rf}")
    print(f"Classification Report:\n{report_rf}")

    # Visualize LSTM and Random Forest Predictions
    visualize_predictions(test.index[lstm_look_back+1:], test['value'].values[lstm_look_back+1:], lstm_predictions_test, y_pred_rf_labels)

    return mse_rf, y_pred_rf_labels


# Function to visualize LSTM and Random Forest Predictions
def visualize_predictions(index, true_values, lstm_predictions, rf_predictions):
    plt.figure(figsize=(15, 6))
    plt.plot(index, true_values, label='True Values', marker='o')
    plt.plot(index, lstm_predictions, label='LSTM Predictions', marker='o')
    plt.plot(index, rf_predictions, label='Random Forest Predictions', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('LSTM and Random Forest Predictions')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Generate synthetic time series data
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    synthetic_data = generate_synthetic_data(start_date, end_date)

    # Set the test_size parameter for time series cross-validation
    test_size = 0.2

    # Set the hyperparameters for LSTM and Random Forest
    lstm_look_back = 5
    lstm_units = 50
    lstm_epochs = 50
    rf_n_estimators = 100
    rf_max_depth = None

    # Integrate LSTM and Random Forest
    mse_rf, y_pred_rf_labels = integrate_lstm_random_forest(synthetic_data['value'], test_size, lstm_look_back,
                                                            lstm_units, lstm_epochs, rf_n_estimators, rf_max_depth)
