
def build_rnn_onehot(input_shape, rnn_units=[64 , 32], rnn_type='LSTM',
                     dense_units=[32 , 16], dropout_rate=0.4, recurrent_dropout=0.2):

    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import regularizers

    if rnn_type.upper() == 'LSTM':
        RNN_Layer = layers.LSTM
    elif rnn_type.upper() == 'GRU':
        RNN_Layer = layers.GRU
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}. Use 'LSTM' or 'GRU'")

    model = keras.Sequential(name=f"{rnn_type}_OneHot")

    model.add(layers.Input(shape=input_shape))

    for i, units in enumerate(rnn_units):
        return_sequences = (i < len(rnn_units) - 1)

        model.add(layers.Bidirectional(
            RNN_Layer(
                units=units,
                return_sequences=return_sequences,
                dropout=recurrent_dropout,
                recurrent_regularizer=regularizers.l2(1e-4),
                name=f'{rnn_type.lower()}_{i+1}'
            ),
            name=f'bidirectional_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_rnn_{i+1}'))

        if return_sequences:
            model.add(layers.Dropout(dropout_rate * 0.5, name=f'dropout_rnn_{i+1}'))

    for i, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(layers.BatchNormalization(name=f'bn_dense_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}'))

    model.add(layers.Dropout(0.3, name='dropout_final'))
    model.add(layers.Dense(1, activation='sigmoid', name='output'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    print("\n\nModel Architecture:\n\n")
    model.summary()
    return model


def build_rnn_kmer(input_shape, rnn_units=[64 , 32], rnn_type='LSTM',
                   dense_units=[32 , 16], dropout_rate=0.4, recurrent_dropout=0.2):

    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import regularizers

    if rnn_type.upper() == 'LSTM':
        RNN_Layer = layers.LSTM
    elif rnn_type.upper() == 'GRU':
        RNN_Layer = layers.GRU
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}. Use 'LSTM' or 'GRU'")

    model = keras.Sequential(name=f"{rnn_type}_Kmer")

    model.add(layers.Input(shape=input_shape))

    if input_shape[1] > 128:
        model.add(layers.TimeDistributed(
            layers.Dense(128, activation='relu'),
            name='embedding_projection'
        ))
        model.add(layers.Dropout(dropout_rate * 0.3, name='dropout_projection'))

    for i, units in enumerate(rnn_units):
        return_sequences = (i < len(rnn_units) - 1)

        model.add(layers.Bidirectional(
            RNN_Layer(
                units=units,
                return_sequences=return_sequences,
                dropout=recurrent_dropout,
                recurrent_regularizer=regularizers.l2(1e-4),
                name=f'{rnn_type.lower()}_{i+1}'
            ),
            name=f'bidirectional_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_rnn_{i+1}'))

        if return_sequences:
            model.add(layers.Dropout(dropout_rate * 0.5, name=f'dropout_rnn_{i+1}'))

    for i, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(layers.BatchNormalization(name=f'bn_dense_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}'))

    model.add(layers.Dropout(0.3, name='dropout_final'))
    model.add(layers.Dense(1, activation='sigmoid', name='output'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    print("\n\nModel Architecture:\n\n")
    model.summary()
    return model
