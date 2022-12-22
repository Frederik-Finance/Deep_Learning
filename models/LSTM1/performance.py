
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import re
import os
from datetime import datetime, timedelta


# # read the data from an Excel file
# data = pd.read_excel(
#     r'/root/deep_learning/data/datasets/PredictionData.xlsx', engine='openpyxl')


# # select all rows except the last 2000 rows
# data = data.iloc[-2000:]


def map_timestamp(datapoint, tf, ahead):
    return int(float(datapoint) + (float(re.sub('\D', '', tf)) * ahead) * 60000)


def _unpickle(filename):
    with open(str(filename), 'rb') as fp:
        return pickle.load(fp)


n_past = 48


def prediction(model, data, ahead=1):
    try:
        # Get latest timestamp
        latest_timestamp = pd.DataFrame(data[-1:]).index[-1]

        latest_timestamp = datetime.fromtimestamp(latest_timestamp / 1000)

        # Calculate prediction timestamp
        print(latest_timestamp)
        prediction_timestamp = latest_timestamp + timedelta(hours=1)

        # Scale and apply PCA to data
        scaler = RobustScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)

        pca = PCA(n_components=0.85)
        pca.fit(data)
        X_pca_new = pca.transform(scaled_data)

        X_pca_new_reshaped = X_pca_new.reshape(-1, n_past, pca.n_components_)

    # Now you can use the reshaped new data with the model to make predictions
        forecast = model.predict(X_pca_new_reshaped)
        forecast_copies = np.repeat(forecast, 27, axis=1)
        y_pred = scaler.inverse_transform(forecast_copies)[:, 0]

        lt = pd.to_datetime(latest_timestamp, unit='ms')

        pt = pd.to_datetime(prediction_timestamp, unit='ms')

        print(
            f'using {lt} to predict the price {y_pred[0]} for {pt}')

        print(y_pred)

        # Delete scaler
        del scaler

        # Create dictionary of prediction data
        prediction_dict = {
            "Time": prediction_timestamp, "Prediction": y_pred[0]}
        return prediction_dict

    except Exception as e:
        print(e)
        pass


def read_excel_to_dataframe_and_create_slice(filename, n):
    # Read the Excel file into a dataframe, using the first column as the index
    df = pd.read_excel(filename, index_col=0)
    print(df)
    df = df  # .iloc[-2250:-1500]

    # Create a slice of the past n entries
    slice_size = df.shape[0] - n
    slice = [df.iloc[i:i+n] for i in range(slice_size)]

    # Return the slice
    return slice


def predictions_to_excel(model, tf, model_input, ahead):
    tf = '5min'

    # Create an empty list to store the prediction data
    prediction_data = []

    # Loop through the model input data
    for data in model_input:
        # Make a prediction using the specified model
        prediction_dict = prediction(model,  data, ahead=ahead)

        # Append the prediction data to the list
        if prediction_dict:
            prediction_data.append(prediction_dict)

    # Create a dataframe from the prediction data
    predictions_df = pd.DataFrame(prediction_data)
    print(predictions_df)

    # Write the dataframe to an Excel file
    # predictions_df.to_excel('./algo/data/predictions_model3_20.xlsx')

    # Return the predictions dataframe
    return predictions_df


model3_1min_20min_120_min = _unpickle('LSTM6_large_data')


model3_1min_20min_120_min_data = read_excel_to_dataframe_and_create_slice(
    r'/root/deep_learning/data/datasets/PredictionData.xlsx', 48)
# print(model3_1min_20min_120_min_data)

# model3_1min_20min_120_min_data = model3_1min_20min_120_min_data.astype(float)
# model3_1min_20min_120_min_data['Time'] = pd.to_datetime(
#     model3_1min_20min_120_min_data['Time'], unit='ms')

model3_1min_20min_120_min_data = model3_1min_20min_120_min_data[-750:]
predictions_df = predictions_to_excel(model3_1min_20min_120_min, 12,
                                      model3_1min_20min_120_min_data, 48)


# Read the Excel file and create a slice of the data
df = pd.read_excel(
    r'/root/deep_learning/data/datasets/PredictionData.xlsx', index_col=0)

combined_df = pd.merge(df, predictions_df, on="Time", how="inner")
# combined_df = combined_df.astype(float)

print(combined_df)

# Plot the 'close_BTC', 'Time', and 'Prediction' columns, with 'Time' as the x-axis
# combined_df[['Close_DOGE', 'Prediction']].plot(x='Time')

combined_df['Time'] = pd.to_datetime(combined_df['Time'], unit='ms')
# combined_df['Error'] = combined_df['Prediction'] - combined_df['Close_DOGE']

#combined_df = combined_df.dropna()

# print(combined_df['Error'])

combined_df.to_excel(r'/root/deep_learning/data/datasets/combined_data.xlsx')


plt.figure(figsize=(30, 20))
combined_df.plot(x="Time", y=["Close_DOGE", "Prediction"],
                 kind="line", color=['y', 'b', 'r'])


# Create list of x-coordinates for 1st and 25th minute of each hour
x_coords = [time for time in combined_df['Time'] if time.minute in [0, 60]]

# Loop through x_coords and draw vertical lines using axvline()
for x in x_coords:
    plt.axvline(x=x, color='k', linestyle='--', linewidth=1)

# Display the chart
plt.show()


# using now() to get current time
now = datetime.now()
plt.savefig(f'/root/deep_learning/data/datasets/chart-{now}.png')

# pd.to_datetime(predictions_df, unit='ms')
# pd.to_datetime(df, unit='ms')
# print(predictions_df.tail(1))
# print(df.tail(1))
