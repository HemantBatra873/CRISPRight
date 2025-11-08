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
