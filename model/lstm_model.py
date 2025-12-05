from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from keras.optimizers import Adam

def build_base_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    return model


def build_advanced_lstm_model(input_shape):
    model = Sequential()

    # 1. CNN Feature Extractor
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # 2. BiLSTM layer (global sequence understanding)
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.15))

    # 3. Stacked LSTM layers
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.1))

    # 4. Dense Head
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_absolute_error'
    )

    return model
