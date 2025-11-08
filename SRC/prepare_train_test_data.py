
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
