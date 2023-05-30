import keras
from keras import Model, Input
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, RepeatVector
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class LSTMAutoencoder(Model):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers, dropout=0.0):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = Sequential()

        self.decoder = Sequential()

        for i in range(num_layers):
            if i == 0:
                self.encoder.add(LSTM(
                    input_shape=(input_size, 1),
                    units=hidden_size,
                    return_sequences=True,
                    dropout=dropout if num_layers > 1 else 0.0
                ))
                self.decoder.add(TimeDistributed(Dense(1, activation='tanh')))
                self.decoder.add(LSTM(
                    input_shape=(embedding_size, 1),
                    units=hidden_size,
                    return_sequences=True,
                    dropout=dropout if num_layers > 1 else 0.0
                ))
            elif i == num_layers - 1:
                self.encoder.add(LSTM(
                    units=embedding_size,
                    return_sequences=True,
                    dropout=dropout if num_layers > 1 else 0.0
                ))
                self.encoder.add(TimeDistributed(Dense(1, activation='tanh')))
                self.decoder.add(LSTM(
                    units=input_size,
                    return_sequences=True,
                    dropout=dropout if num_layers > 1 else 0.0
                ))
                self.decoder.add(TimeDistributed(Dense(1, activation='tanh')))
            else:
                self.encoder.add(LSTM(
                    units=hidden_size,
                    return_sequences=True,
                    dropout=dropout if num_layers > 1 else 0.0
                ))
                self.decoder.add(LSTM(
                    units=hidden_size,
                    return_sequences=True,
                    dropout=dropout if num_layers > 1 else 0.0
                ))

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


class PlotRandomSampleCallback(Callback):
    def __init__(self, data, plot_every_n_epochs=10):
        super().__init__()
        self.data = data
        self.plot_every_n_epochs = plot_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.plot_every_n_epochs == 0:
            idx = np.random.randint(len(self.data))
            sample_input = self.data[idx:idx + 1]  # Get a random sample input
            sample_output = self.model.predict(sample_input)  # Get the output for the sample input

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"Input at Epoch {epoch + 1}")
            plt.plot(sample_input[0])

            plt.subplot(1, 2, 2)
            plt.title(f"Output at Epoch {epoch + 1}")
            plt.plot(sample_output[0])

            plt.show()
