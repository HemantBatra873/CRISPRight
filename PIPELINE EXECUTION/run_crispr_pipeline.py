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
