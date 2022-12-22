
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


# read the data from an Excel file
data = pd.read_excel(
    r'/root/deep_learning/data/datasets/PredictionData.xlsx', engine='openpyxl')


def _pickle(filename, lst):
    with open(str(filename), 'wb') as fp:
        pickle.dump(lst, fp)


print(data)
# select all rows except the last 2000 rows
data = data.iloc[-24000:-2000]

# columns used in training
cols = list(data)[1:28]
train_data = data[cols].astype(float)

scaler = RobustScaler()
scaler.fit(train_data)
scaled_data = scaler.transform(train_data)

# Apply PCA to the scaled data
pca = PCA(n_components=0.85)
X_pca = pca.fit_transform(scaled_data)

print(scaled_data)
print(X_pca)

trainX, trainY = [], []

# for now only one in the future
# later 1 - 15 in the future
n_future = 12
n_past = 48

batch_size = 16

for i in range(n_past, len(X_pca) - n_future + 1, batch_size):
    trainX.append(X_pca[i - n_past:i, 0:X_pca.shape[1]])
    trainY.append(scaled_data[i+n_future - 1:i+n_future, 0])


trainX, trainY = np.array(trainX), np.array(trainY)

# split the data into training, validation, and test sets
train_split = 0.8
val_split = 0.1
test_split = 0.1

train_len = int(len(trainX) * train_split)
val_len = int(len(trainX) * val_split)
test_len = len(trainX) - train_len - val_len
trainX_, valX, testX = trainX[:train_len], trainX[train_len:train_len +
                                                  val_len], trainX[train_len+val_len:]
trainY_, valY, testY = trainY[:train_len], trainY[train_len:train_len +
                                                  val_len], trainY[train_len+val_len:]


# Build the model
model = Sequential()

print(trainX.shape)

# Add LSTM layers with 128 units and dropout regularization
model.add(LSTM(128, input_shape=(
    n_past, pca.n_components_), return_sequences=True, dropout=0.2))
model.add(LSTM(128, return_sequences=True, dropout=0.2))
model.add(LSTM(128, return_sequences=False, dropout=0.2))

# model.add(Lambda(lambda x: np.log(x + 1)))

model.add(Dense(32, activation='sigmoid'))

# Add a fully connected layer with 32 units and dropout regularization
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.3))

# Add an output layer with a single unit
model.add(Dense(1))

# Compile the model using the Adam optimizer
model.compile(optimizer='adam', loss='mse')


# use early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
history = model.fit(trainX, trainY, epochs=80, batch_size=16,
                    validation_data=(valX, valY), callbacks=[early_stop],
                    verbose=1)


_pickle('LSTMX_short_term_PCA_mse', model)
model.save('./path/to/save/model.h5')
model.save_weights('./path/to/save/model_weights.h5')


def plot_history(history):
    # get the training and validation loss
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    # plot the losses
    plt.plot(val_loss, label='val_loss')
    plt.plot(train_loss, label='train_loss')
    plt.legend()
    plt.show()

    # save the figure
    plt.savefig('model_loss.png')


# plot the history of the model after training
plot_history(history)


# val_predictions = model.predict(valX)
# test_predictions = model.predict(testX)

# # print the MSE of the predictions on the validation and test data
# val_mse = keras.losses.mean_squared_error(valY, val_predictions).numpy()
# test_mse = keras.losses.mean_squared_error(testY, test_predictions).numpy()

# print(f'Validation MSE: {val_mse:.3f}')
# print(f'Test MSE: {test_mse:.3f}')


# # un-scale the data for plotting
# valY_plot = scaler.inverse_transform(valY.reshape(-1, 1))
# val_predictions_plot = scaler.inverse_transform(val_predictions.reshape(-1, 1))

# # plot the actual and predicted values
# plt.plot(valY_plot, label='Actual')
# plt.plot(val_predictions_plot, label='Predicted')
# plt.legend()
# plt.show()

# plt.savefig('Predictions.png')


# # get predictions from the model
# predictions = model.predict(testX)

# # inverse transform the predictions using the scaler object
# predictions = scaler.inverse_transform(predictions)

# # inverse transform the actual values using the scaler object
# testY = scaler.inverse_transform(testY)

# # apply inverse PCA transformation to the predictions
# predictions = pca.inverse_transform(predictions)


# get predictions from the model
predictions = model.predict(testX)


predictions = np.repeat(predictions, 24, axis=1)
# inverse transform the predictions using the scaler object
predictions = scaler.inverse_transform(predictions)[:, 0]

# # inverse transform the actual values using the scaler object
# testY = np.repeat(predictions, 24, axis=1)
# testY = scaler.inverse_transform(testY)

# predictions = np.repeat(predictions, 2, axis=1)


# apply inverse PCA transformation to the predictions
#predictions = pca.inverse_transform(predictions)[:, 0]
print(predictions)


plt.plot(predictions, label='Predictions')
plt.plot(testY, label='Actual')
plt.legend()
plt.show()
plt.savefig('predictions_MF')
