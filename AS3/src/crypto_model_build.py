import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tqdm import tqdm


class ModelBuilder:
    # Hyper parameters
    def __init__(self, path_excel, model_name, load=False):
        self.SEQ_LEN = 100
        self.TRAIN_SPLIT = 0.9  # Set training and testing part in ratio 9:1
        self.BATCH_SIZE = 64  # Set Batch size
        self.EPOCHS = 50  # Set 50 epochs for traning model
        self.VALIDATION_SPLIT = 0.15

        self.path_in = path_excel
        self.path_out = '{}/saved_model/{}'.format(os.getcwd(), model_name)
        self.loaded = load
        self.model = Sequential() if not self.loaded else load_model(self.path_out)

        self.scaler = MinMaxScaler()
        self.training_data = None
        self.history, self.df = None, None
        self.X_train, self.Y_train, self.X_test, self.Y_test = None, None, None, None
        self.loss, self.acc = None, None

    def open_excel(self) -> pd.DataFrame:
        return pd.read_excel(self.path_in)

    def _normalization(self):
        data = to_sequences(self.training_data, self.SEQ_LEN)
        num_train = int(self.TRAIN_SPLIT * data.shape[0])

        self.X_train = data[:num_train, :-1, :]
        self.Y_train = data[:num_train, -1, :]
        self.X_test = data[num_train:, :-1, :]
        self.Y_test = data[num_train:, -1, :]

    def _create_model(self):
        self.model.add(Bidirectional(LSTM(self.SEQ_LEN - 1, return_sequences=True),
                                     input_shape=(self.SEQ_LEN - 1, self.X_train.shape[-1])))
        self.model.add(Dropout(rate=0.4))
        self.model.add(Bidirectional(LSTM((self.SEQ_LEN - 1) * 2, return_sequences=True)))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Bidirectional(LSTM(self.SEQ_LEN - 1, return_sequences=False)))
        self.model.add(Dense(units=6))
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])

    def build_model(self):
        self.df = self.open_excel()
        self.training_data = self.scaler.fit_transform(self.df.drop(["Open time"], axis=1).fillna(0))
        self._normalization()
        self._create_model()
        self.model.summary()

    def train_model(self):
        self.df = self.open_excel()
        self.training_data = self.scaler.fit_transform(self.df.drop(["Open time"], axis=1).fillna(0))
        self._normalization()
        self.history = self.model.fit(self.X_train,
                                      self.Y_train,
                                      epochs=self.EPOCHS,
                                      batch_size=self.BATCH_SIZE,
                                      validation_split=self.VALIDATION_SPLIT)

    def evaluate_model(self, assets_dir, pair):
        self.loss, self.acc = self.model.evaluate(self.X_test, self.Y_test)

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        pair_dir = os.path.join(assets_dir, pair)

        if not os.path.exists(pair_dir):
            os.mkdir(pair_dir)
        loss_image = os.path.join(pair_dir, 'Model_Loss_{}.png'.format(pair))
        plt.savefig(loss_image)
        plt.clf()

        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        if not os.path.exists(pair_dir):
            os.mkdir(pair_dir)
        accuracy_image = os.path.join(pair_dir, 'Model_Accuracy_{}.png'.format(pair))
        plt.savefig(accuracy_image)
        plt.clf()

    def test_prediction(self, assets_dir, pair):
        y_pred = self.model.predict(self.X_test)

        y_test_inverse = self.scaler.inverse_transform(self.Y_test)
        y_pred_inverse = self.scaler.inverse_transform(y_pred)

        plt.plot(y_test_inverse[:, 0], label="Actual Price", color='green')
        plt.plot(y_pred_inverse[:, 0], label="Predicted Price", color='red')

        plt.title('Binance Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='best')

        # plt.show()
        pair_dir = os.path.join(assets_dir, pair)
        if not os.path.exists(pair_dir):
            os.mkdir(pair_dir)
        predict_image = os.path.join(pair_dir, 'Price_Prediction_{}.png'.format(pair))
        plt.savefig(predict_image)
        plt.clf()

    def future_prediction(self, assets_dir, pair):
        X_pred = self.X_test
        time = 24 * 10
        for _ in tqdm(range(time)):
            X_pred = np.vstack(
                [X_pred, np.expand_dims(np.vstack([X_pred[-1, 1:, :], self.model.predict(X_pred)[-1]]), axis=0)])
        y_pred_inv = self.scaler.inverse_transform(X_pred[-time:, -1, :])
        y = np.vstack([self.scaler.inverse_transform(self.model.predict(self.X_test)), y_pred_inv])

        plt.plot(self.scaler.inverse_transform(self.Y_test)[:, 0], label="Actual price", color='green')
        plt.plot(y[:, 0], label="Predicted Price", color='red')

        plt.title('Future Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='best')

        # plt.show()
        pair_dir = os.path.join(assets_dir, pair)
        if not os.path.exists(pair_dir):
            os.mkdir(pair_dir)
        predict_image = os.path.join(pair_dir, 'Future_Price_Prediction_{}.png'.format(pair))
        plt.savefig(predict_image)
        plt.clf()

    def save_model(self):
        if not self.loaded:
            self.model.save(self.path_out)


def to_sequences(data, seq_len) -> np.array:
    return np.array([data[i:i + seq_len] for i in range(len(data) - seq_len)])
