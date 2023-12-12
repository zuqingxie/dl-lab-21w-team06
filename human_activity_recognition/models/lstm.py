import gin
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN


@gin.configurable
def rnn_mix(window_size, dropout_rate, units, dense_units):
    """
    Define a RNN_MIX model with combination RNN layers: LSTM, GRU, SimpleRNN
    Args:
        window_size: window size of input
        dropout_rate: deopout rate
        units: Dimensionality of the output.
        dense_units: number of the dense units

    Returns:
        (keras.Model): keras model object
    """
    model = Sequential()
    model.add(LSTM(units, input_shape=(window_size, 6), return_sequences=True))
    for _ in range(1):
        model.add(LSTM(units, return_sequences=True))
        model.add(GRU(units, return_sequences=True))
        model.add(SimpleRNN(units,  return_sequences=True))
    model.add(SimpleRNN(units,  return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    return model


@gin.configurable
def rnn(window_size, lstm_units, dropout_rate, dense_units, rnn_name, num_rnn, activation):
    """
    Define RNN model
    Args:
        window_size: window size of input
        lstm_units: Dimensionality of the output.
        dropout_rate: dropout rate
        dense_units: number of the dense units
        rnn_name: lstm, gru or simple_rnn
        num_rnn: number of rnn layer
        activation: activity functiton

    Returns:
        (keras.Model): keras model object
    """
    assert num_rnn > 2, 'Number of rnn has to be at least 1.'

    model = Sequential(name=rnn_name)

    if rnn_name == "lstm":
        model.add(LSTM(lstm_units, input_shape=(window_size, 6), return_sequences=True))
        for _ in range(num_rnn - 2):
            model.add(LSTM(lstm_units, activation=activation, return_sequences=True))
        model.add(LSTM(lstm_units, return_sequences=False))
    elif rnn_name == "gru":
        model.add(GRU(lstm_units, input_shape=(window_size, 6), return_sequences=True))
        for _ in range(num_rnn - 2):
            model.add(GRU(lstm_units, activation=activation, return_sequences=True))
        model.add(GRU(lstm_units, return_sequences=False))
    elif rnn_name == "simple_rnn":
        model.add(SimpleRNN(lstm_units, input_shape=(window_size, 6), return_sequences=True))
        for _ in range(num_rnn - 2):
            model.add(SimpleRNN(lstm_units, activation=activation, return_sequences=True))
        model.add(SimpleRNN(lstm_units, return_sequences=False))
    else:
        raise ValueError

    model.add(Dense(dense_units, kernel_regularizer=regularizers.L1L2(l1=0.01, l2=0.005)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(12, activation='softmax'))

    return model
