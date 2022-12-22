
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
    r'/root/deep_learning/data/datasets/Features.xlsx', engine='openpyxl')


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

model = Sequential()

# Add multiple LSTM layers with 256 units and dropout regularization
model.add(LSTM(256, input_shape=(n_past, pca.n_components_),
          return_sequences=True, dropout=0.2))
model.add(LSTM(256, return_sequences=True, dropout=0.2))
model.add(LSTM(256, return_sequences=True, dropout=0.2))
model.add(LSTM(256, return_sequences=False, dropout=0.2))

# Add multiple fully connected layers with 512 units and dropout regularization
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.3))

model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')

# use early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
history = model.fit(trainX, trainY, epochs=80, batch_size=16,
                    validation_data=(valX, valY), callbacks=[early_stop],
                    verbose=1)


_pickle('LSTM6_large_data', model)
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
