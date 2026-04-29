def load_dataset(input_file, output_file, mount_drive=True):

    import pandas as pd

    if mount_drive:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")

    print(f"Reading file from: {input_file}")
    df = pd.read_csv(input_file, sep='\t',low_memory=False)

    df.to_csv(output_file, index=False)
    print(f"File successfully converted and saved to: {output_file}")
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):

    import pandas as pd
    import numpy as np
    from sklearn.utils import resample

    print("Selecting relevant columns...")
    required_cols = ['Guide_sequence', 'Target_sequence', 'PAM', 'Identity']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy()
    print(f"Selected columns: {required_cols}")
    print(f"Shape after selection: {df.shape}")

    print("Handling missing values...")
    df = df.dropna()

    print("Standardizing sequences...")
    df['Guide_sequence'] = df['Guide_sequence'].str.upper().str.strip()
    df['Target_sequence'] = df['Target_sequence'].str.upper().str.strip()
    df['PAM'] = df['PAM'].str.upper().str.strip()
    df['Identity'] = df['Identity'].str.upper().str.strip()

    print("Validating sequence composition...")
    valid_bases = set('ATGC')

    def is_valid_sequence(seq):
        return set(seq).issubset(valid_bases) and len(seq) > 0

    valid_guide = df['Guide_sequence'].apply(is_valid_sequence)
    valid_target = df['Target_sequence'].apply(is_valid_sequence)
    valid_pam = df['PAM'].apply(is_valid_sequence)

    invalid_count = (~(valid_guide & valid_target & valid_pam)).sum()
    print(f"Invalid sequences found: {invalid_count}")

    df = df[valid_guide & valid_target & valid_pam].copy()
    print(f"Shape after validation: {df.shape}")

    print("Analyzing sequence lengths...")
    guide_lengths = df['Guide_sequence'].str.len()
    target_lengths = df['Target_sequence'].str.len()
    pam_lengths = df['PAM'].str.len()

    print(f"Guide sequence length: {guide_lengths.min()}-{guide_lengths.max()} (mode: {guide_lengths.mode()[0]})")
    print(f"Target sequence length: {target_lengths.min()}-{target_lengths.max()} (mode: {target_lengths.mode()[0]})")
    print(f"PAM length: {pam_lengths.min()}-{pam_lengths.max()} (mode: {pam_lengths.mode()[0]})")

    print("\nConverting Identity to binary labels...")
    print(f"Identity value distribution:\n{df['Identity'].value_counts()}")
    df = df[df['Identity'].isin(['ON', 'OFF'])].copy()
    df['label'] = (df['Identity'].str.upper() == 'ON').astype(int)


    df_on = df[df['label'] == 1]
    df_off = df[df['label'] == 0]
    desired_off_count = min(len(df_on) * 4, len(df_off))
    df_off_downsampled = resample(
        df_off,
        replace=False,
        n_samples=desired_off_count,
        random_state=42
    )
    df_balanced = pd.concat([df_on , df_off_downsampled], axis=0).sample(frac=1, random_state=42)
    df = df_balanced.copy()


    print("Creating combined sequence...")
    df['combined_sequence'] = df['Guide_sequence'] + df['Target_sequence'] + df['PAM']
    df['combined_sequence'] = df['combined_sequence'].str[:49]
    df = df.reset_index(drop=True)
    print("PREPROCESSING COMPLETE")
    print(f"Final columns: {list(df.columns)}")

    return df

def prepare_train_test(df, encoding="onehot", test_size=0.2, random_state=42, k=3):

    import numpy as np
    from sklearn.model_selection import train_test_split
    from collections import Counter

    sequences = df['combined_sequence'].values
    labels = df['label'].values

    print(f"Total samples: {len(sequences)}")
    print(f"Sequence length: {len(sequences[0])}")

    if encoding == "onehot":
        print("\n\nApplying One-Hot Encoding...")

        base_to_index = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        seq_length = len(sequences[0])
        n_samples = len(sequences)
        X = np.zeros((n_samples, seq_length, 4), dtype=np.float32)

        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq):
                if base in base_to_index:
                    X[i, j, base_to_index[base]] = 1.0

        print(f"Encoded shape: {X.shape}")

    elif encoding == "kmer":
        print(f"\n\nApplying K-mer Encoding (k={k})...")

        def sequence_to_kmers(seq, k):
            """Extract k-mers from sequence"""
            kmers = []
            for i in range(len(seq) - k + 1):
                kmers.append(seq[i:i+k])
            return kmers

        all_kmers = []
        for seq in sequences:
            all_kmers.extend(sequence_to_kmers(seq, k))

        kmer_counts = Counter(all_kmers)
        vocab = sorted(kmer_counts.keys())
        kmer_to_index = {kmer: idx for idx, kmer in enumerate(vocab)}

        print(f"Vocabulary size: {len(vocab)}")
        print(f"Most common k-mers: {kmer_counts.most_common(10)}")

        n_samples = len(sequences)
        seq_length = len(sequences[0])
        n_kmers = seq_length - k + 1
        vocab_size = len(vocab)

        X = np.zeros((n_samples, n_kmers, vocab_size), dtype=np.float32)

        for i, seq in enumerate(sequences):
            kmers = sequence_to_kmers(seq, k)
            for j, kmer in enumerate(kmers):
                if kmer in kmer_to_index:
                    X[i, j, kmer_to_index[kmer]] = 1.0

        print(f"Encoded shape: {X.shape}")

    else:
        raise ValueError(f"Unknown encoding: {encoding}. Use 'onehot' or 'kmer'")

    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train labels - On-target: {np.sum(y_train)}, Off-target: {len(y_train) - np.sum(y_train)}")
    print(f"Test labels - On-target: {np.sum(y_test)}, Off-target: {len(y_test) - np.sum(y_test)}")
    return X_train, X_test, y_train, y_test

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

def train_model(model, X_train, y_train, X_test, y_test,
                epochs=50, batch_size=32, verbose=1):

    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    import matplotlib.pyplot as plt

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

def evaluate_model(model, X_test, y_test, threshold=0.5):

    import numpy as np

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve, average_precision_score
    )

    print(f"\n\n\nEVALUATING MODEL: {model.name}")

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall

    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)

    print("\n\nCLASSIFICATION METRICS\n")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"AUC-ROC:      {auc_roc:.4f}")
    print(f"AUC-PR:       {auc_pr:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")

    print("\n\nCONFUSION MATRIX\n")
    print(f"                  Predicted")
    print(f"                  Neg    Pos")
    print(f"Actual  Neg     {tn:>5}  {fp:>5}")
    print(f"        Pos     {fn:>5}  {tp:>5}")

    print("\n\nDETAILED METRICS\n")
    print(f"True Positives:   {tp}")
    print(f"True Negatives:   {tn}")
    print(f"False Positives:  {fp}")
    print(f"False Negatives:  {fn}")
    print(f"Total Samples:    {len(y_test)}")

    print("\n\nCLASSIFICATION REPORT\n")
    print(classification_report(y_test, y_pred,
                                target_names=['Off-target', 'On-target'],
                                digits=4))

    results = {
        'model_name': model.name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds_roc},
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=['Off-target', 'On-target'],
            output_dict=True
        )
    }

    return results

def plot_evaluation_results(results):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    import warnings
    warnings.filterwarnings("ignore")

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (16, 10)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

    model_name = results.get("model_name", "Model")

    y_true = results["y_true"]
    y_pred = results["y_pred"]
    y_pred_proba = results["y_pred_proba"]
    cm = results["confusion_matrix"]
    fpr = results["roc_curve"]["fpr"]
    tpr = results["roc_curve"]["tpr"]
    auc_roc = results["auc_roc"]
    auc_pr = results["auc_pr"]
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    metrics = {
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1-Score": results["f1_score"],
        "AUC-ROC": results["auc_roc"],
        "AUC-PR": results["auc_pr"],
        "Sensitivity": results["sensitivity"],
        "Specificity": results["specificity"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Evaluation Summary for {model_name}", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {auc_roc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {auc_pr:.3f}")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Off-target", "On-target"],
                yticklabels=["Off-target", "On-target"],
                cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax = axes[1, 0]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    sns.barplot(x=metric_names, y=metric_values, hue=metric_names, palette="viridis", legend=False, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Performance Metrics")
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    ax = axes[1, 1]
    sns.histplot(y_pred_proba[y_true == 0], bins=30, color="skyblue", label="Off-target", kde=True, ax=ax)
    sns.histplot(y_pred_proba[y_true == 1], bins=30, color="orange", label="On-target", kde=True, ax=ax)
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted Probability (y_pred_proba)")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1, 2]
    report = results["classification_report"]
    report_df = (
        sns.heatmap(
            np.array([
                [report["Off-target"]["precision"], report["Off-target"]["recall"], report["Off-target"]["f1-score"]],
                [report["On-target"]["precision"], report["On-target"]["recall"], report["On-target"]["f1-score"]],
            ]),
            annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=["Precision", "Recall", "F1-score"],
            yticklabels=["Off-target", "On-target"],
            cbar=False, ax=ax
        )
    )
    ax.set_title("Class-wise Metrics (from classification report)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print(f"\n\nPlots generated for {model_name}")

def compare_models(results):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n\n\nMODEL COMPARISON ANALYSIS")

    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'AUC-ROC': result['auc_roc'],
            'AUC-PR': result['auc_pr'],
            'Sensitivity': result['sensitivity'],
            'Specificity': result['specificity'],
            'TP': result['true_positives'],
            'TN': result['true_negatives'],
            'FP': result['false_positives'],
            'FN': result['false_negatives']
        })

    df_comparison = pd.DataFrame(comparison_data)

    print("\nPERFORMANCE METRICS SUMMARY")
    print(df_comparison.to_string(index=False))

    print("\nBEST PERFORMERS BY METRIC")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC',
               'AUC-PR', 'Sensitivity', 'Specificity']

    for metric in metrics:
        best_idx = df_comparison[metric].idxmax()
        best_model = df_comparison.loc[best_idx, 'Model']
        best_value = df_comparison.loc[best_idx, metric]
        print(f"{metric:15s}: {best_model:20s} ({best_value:.4f})")

    print("\n\nOVERALL MODEL RANKING")

    metrics_to_rank = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    df_normalized = df_comparison[metrics_to_rank].copy()

    df_comparison['Overall_Score'] = df_normalized.mean(axis=1)
    df_ranked = df_comparison.sort_values('Overall_Score', ascending=False)

    print("\nRank Model Overall Score")
    for idx, (_, row) in enumerate(df_ranked.iterrows(), 1):
        print(f"{idx:2d}.   {row['Model']:20s}     {row['Overall_Score']:.4f}")

    print("\nSTATISTICAL SUMMARY")

    stats_df = df_comparison[metrics].describe().T
    stats_df = stats_df[['mean', 'std', 'min', 'max']]
    print(stats_df.to_string())

    print("\nCONFUSION MATRIX ANALYSIS\n")

    for model_name, result in results.items():
        cm = result['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        print(f"\n{model_name}:")
        print(f"  True Negatives:  {tn:5d} ({tn/total*100:5.2f}%)")
        print(f"  False Positives: {fp:5d} ({fp/total*100:5.2f}%)")
        print(f"  False Negatives: {fn:5d} ({fn/total*100:5.2f}%)")
        print(f"  True Positives:  {tp:5d} ({tp/total*100:5.2f}%)")

    print("\nGENERATING COMPARISON VISUALIZATIONS")

    fig = plt.figure(figsize=(20, 12))
    ax1 = plt.subplot(2, 3, 1, projection='polar')

    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    colors_radar = plt.cm.Set3(np.linspace(0, 1, len(results)))

    for idx, (model_name, result) in enumerate(results.items()):
        values = [result['accuracy'], result['precision'], result['recall'],
                  result['f1_score'], result['auc_roc']]
        values += values[:1]

        ax1.plot(angles, values, 'o-', linewidth=2, label=model_name,
                color=colors_radar[idx])
        ax1.fill(angles, values, alpha=0.15, color=colors_radar[idx])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=10)
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Performance Radar Chart', size=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax1.grid(True)

    ax2 = plt.subplot(2, 3, 2)

    x = np.arange(len(metrics_to_rank))
    width = 0.8 / len(results)
    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for idx, (model_name, result) in enumerate(results.items()):
        values = [result['accuracy'], result['precision'], result['recall'],
                  result['f1_score'], result['auc_roc']]
        offset = (idx - len(results)/2) * width + width/2
        ax2.bar(x + offset, values, width, label=model_name, color=colors_bar[idx])

    ax2.set_xlabel('Metrics', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_to_rank, rotation=45, ha='right')
    ax2.legend(fontsize=9)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = plt.subplot(2, 3, 3)

    heatmap_data = df_comparison.set_index('Model')[metrics].T
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                cbar_kws={'label': 'Score'}, ax=ax3, vmin=0, vmax=1)
    ax3.set_title('Metrics Heatmap', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Metric', fontsize=12)

    ax4 = plt.subplot(2, 3, 4)

    cm_categories = ['TN', 'FP', 'FN', 'TP']
    x_cm = np.arange(len(cm_categories))

    for idx, (model_name, result) in enumerate(results.items()):
        values = [result['true_negatives'], result['false_positives'],
                  result['false_negatives'], result['true_positives']]
        offset = (idx - len(results)/2) * width + width/2
        ax4.bar(x_cm + offset, values, width, label=model_name, color=colors_bar[idx])

    ax4.set_xlabel('Prediction Type', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Confusion Matrix Components', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_cm)
    ax4.set_xticklabels(cm_categories)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = plt.subplot(2, 3, 5)

    for idx, (model_name, result) in enumerate(results.items()):
        ax5.scatter(result['specificity'], result['sensitivity'],
                   s=200, alpha=0.6, color=colors_bar[idx],
                   label=model_name, edgecolors='black', linewidth=1.5)

    ax5.set_xlabel('Specificity', fontsize=12)
    ax5.set_ylabel('Sensitivity', fontsize=12)
    ax5.set_title('Sensitivity vs Specificity', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 1.05])
    ax5.set_ylim([0, 1.05])
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)

    df_ranked_plot = df_ranked.sort_values('Overall_Score')
    colors_rank = plt.cm.RdYlGn(df_ranked_plot['Overall_Score'])

    bars = ax6.barh(df_ranked_plot['Model'], df_ranked_plot['Overall_Score'],
                    color=colors_rank, edgecolor='black', linewidth=1.5)
    ax6.set_xlabel('Overall Score', fontsize=12)
    ax6.set_title('Overall Model Ranking', fontsize=14, fontweight='bold')
    ax6.set_xlim([0, 1])
    ax6.grid(True, alpha=0.3, axis='x')

    for bar in bars:
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("\n\nComparison visualizations generated!\n")

    best_model = df_ranked.iloc[0]['Model']
    print(f"BEST OVERALL MODEL: {best_model}")
    print(f"Overall Score: {df_ranked.iloc[0]['Overall_Score']:.4f}")
    return df_comparison, best_model

def run_crispr_pipeline(input_file, output_file, test_size, random_state, k, epochs, batch_size, rnn_type, mount_drive):

    print("CRISPR-CAS9 OFF-TARGET PREDICTION PIPELINE")

    results, models = {}, {}

    print("\n\n\nLOADING DATASET")
    df = load_dataset(input_file, output_file, mount_drive)

    print("\n\n\nPREPROCESSING DATA")
    df = preprocess_data(df)

    print("\n\n\nPREPARING ONE-HOT ENCODED DATA")
    X_train_1h, X_test_1h, y_train, y_test = prepare_train_test(
        df, encoding="onehot", test_size=test_size, random_state=random_state
    )

    print("STEP 4: TRAINING CNN MODEL (ONE-HOT)")
    cnn_onehot = build_cnn_onehot(X_train_1h.shape[1:])
    cnn_onehot = train_model(cnn_onehot, X_train_1h, y_train, X_test_1h, y_test,
                             epochs=epochs, batch_size=batch_size)
    cnn_onehot_res = evaluate_model(cnn_onehot, X_test_1h, y_test)
    plot_evaluation_results(cnn_onehot_res)
    models['CNN-OneHot'] = cnn_onehot
    results['CNN-OneHot'] = cnn_onehot_res

    print(f"\n\n\nTRAINING RNN-GRU MODEL (ONE-HOT)")
    rnn_onehot = build_rnn_onehot(X_train_1h.shape[1:], rnn_type=rnn_type)
    rnn_onehot = train_model(rnn_onehot, X_train_1h, y_train, X_test_1h, y_test,
                             epochs=epochs, batch_size=batch_size)
    rnn_onehot_res = evaluate_model(rnn_onehot, X_test_1h, y_test)
    plot_evaluation_results(rnn_onehot_res)
    models[f"{rnn_type}-OneHot"] = rnn_onehot
    results[f"{rnn_type}-OneHot"] = rnn_onehot_res

    print("\n\n\nPREPARING K-MER ENCODED DATA")
    X_train_kmer, X_test_kmer, y_train_kmer, y_test_kmer = prepare_train_test(
        df, encoding="kmer", test_size=test_size, random_state=random_state, k=k
    )
    input_shape_kmer = (X_train_kmer.shape[1], X_train_kmer.shape[2])

    print("\nTraining CNN with K-mer encoding...\n")
    cnn_kmer_model = build_cnn_kmer(input_shape_kmer)
    cnn_kmer_model = train_model(
        cnn_kmer_model,
        X_train_kmer, y_train_kmer,
        X_test_kmer, y_test_kmer,
        epochs=epochs, batch_size=batch_size
    )
    cnn_kmer_res = evaluate_model(cnn_kmer_model, X_test_kmer, y_test_kmer)
    plot_evaluation_results(cnn_kmer_res)
    models["CNN-Kmer"] = cnn_kmer_model
    results["CNN-Kmer"] = cnn_kmer_res

    print(f"\n\nTRAINING RNN-BiLSTM MODEL (K-MER)")
    rnn_kmer = build_rnn_kmer(input_shape_kmer, rnn_type=rnn_type)
    rnn_kmer = train_model(
        rnn_kmer,
        X_train_kmer, y_train_kmer,
        X_test_kmer, y_test_kmer,
        epochs=epochs, batch_size=batch_size
    )
    rnn_kmer_res = evaluate_model(rnn_kmer, X_test_kmer, y_test_kmer)
    plot_evaluation_results(rnn_kmer_res)
    models[f"{rnn_type}-Kmer"] = rnn_kmer
    results[f"{rnn_type}-Kmer"] = rnn_kmer_res

    comparison_df, best_model = compare_models(results)

    print("\n\n\nPIPELINE EXECUTION COMPLETE\n")
    print(f"Best Model: {best_model}")

    return results, best_model, models, comparison_df

if __name__ == "__main__":

    results, best_model, models, comparison_df = run_crispr_pipeline(
        input_file='/content/drive/MyDrive/CRISPRight/CRISPRight.txt',
        output_file='/content/drive/MyDrive/CRISPRight/CRISPRight.csv',
        test_size=0.3,
        random_state=42,
        k=4,
        epochs=40,
        batch_size=32,
        rnn_type='LSTM',
        mount_drive=True
    )

    print("\n\n\nSAVING RESULTS")
    comparison_df.to_csv('/content/drive/MyDrive/CRISPRight/model_comparison.csv',
                         index=False)

    print("\n\n\nFINAL SUMMARY")
    print(f"Total Models Trained: {len(models)}")
    print(f"Best Performing Model: {best_model}")
    print(f"Pipeline Finished")
