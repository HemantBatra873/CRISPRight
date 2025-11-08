
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
    desired_off_count = min(len(df_on) * 8, len(df_off))
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
