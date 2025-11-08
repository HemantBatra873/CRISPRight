
def build_cnn_onehot(input_shape, filters=[32, 64, 128], kernel_sizes=[3, 3 , 5],
                     dense_units=[64 , 32], dropout_rate=0.4):

    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import regularizers
    model = keras.Sequential(name="CNN_OneHot")
    model.add(layers.Input(shape=input_shape))

    for i, (n_filters, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        model.add(layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(1e-4),
            name=f'conv1d_{i+1}'
        ))

        if i < 2:
          model.add(layers.BatchNormalization(name=f'bn_{i+1}'))

        model.add(layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}'))
        model.add(layers.Dropout(dropout_rate * 0.5, name=f'dropout_conv_{i+1}'))

    model.add(layers.Flatten(name='flatten'))

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


def build_cnn_kmer(input_shape, filters=[32, 64, 128], kernel_sizes=[3, 5, 5],
                   dense_units=[64, 32], dropout_rate=0.4):

    from tensorflow import keras
    from tensorflow.keras import regularizers
    from tensorflow.keras import layers
    model = keras.Sequential(name="CNN_Kmer")
    model.add(layers.Input(shape=input_shape))

    for i, (n_filters, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        model.add(layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(1e-4),
            name=f'conv1d_{i+1}'
        ))

        if i < 2:
          model.add(layers.BatchNormalization(name=f'bn_{i+1}'))

        model.add(layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}'))
        model.add(layers.Dropout(dropout_rate * 0.5, name=f'dropout_conv_{i+1}'))

    model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))

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
