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
    comparison_df.to_csv('/content/drive/MyDrive/CRISPRight/model_comparison.csv', index=False)
    print("\n\n\nFINAL SUMMARY")
    print(f"Total Models Trained: {len(models)}")
    print(f"Best Performing Model: {best_model}")
    print(f"Pipeline Finished")
