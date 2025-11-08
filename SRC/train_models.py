def train_model(model, X_train, y_train, X_test, y_test,
                epochs=50, batch_size=32, verbose=1):

    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    print(f"\n\n\nTRAINING MODEL: {model.name}")


    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    print(f"\nComputed class weights: {class_weights_dict}\n")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose,
        class_weight=class_weights_dict,
        shuffle=True
    )

    print("\n\nTraining Complete!\n\n\n")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model.name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model.name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    print(f"\n\n\nBest validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Best validation AUC: {max(history.history['val_auc']):.4f}")

    model.history = history

    return model
