import pandas as pd
import numpy as np
import os

def create_random_missing_dataset(
    input_csv="CMU_MOSEI_Text_Labels.csv",
    output_csv="CMU_MOSEI_Random_Missing_Modalities.csv",
    mask_token="",  # Use "" or "[MISSING]"
    seed=42
):
    print("=" * 70)
    print("CREATING RANDOM MISSING MODALITY DATASET (NO PATTERN / MCAR)")
    print("=" * 70)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # Load original dataset
    df = pd.read_csv(input_csv)
    total_samples = len(df)
    print(f"Loaded {total_samples} samples from {input_csv}")

    # Fix corrupted '#NAME?' video ID in row 0 if present (caused by Excel interpreting '--qXJuDtHPw')
    if df.loc[0, "Video_ID"] == "#NAME?":
        df.loc[0, "Video_ID"] = "--qXJuDtHPw"
        print("Fixed corrupted Video_ID in row 0 -> '--qXJuDtHPw'")

    # Set random seed for reproducible randomness
    np.random.seed(seed)

    # The 7 valid non-empty combinations of (Text, Audio, Visual):
    # (1, 1, 1) -> Complete (All modalities present)
    # (1, 1, 0) -> Text + Audio (Visual missing)
    # (1, 0, 1) -> Text + Visual (Audio missing)
    # (0, 1, 1) -> Audio + Visual (Text missing)
    # (1, 0, 0) -> Text only (Audio & Visual missing)
    # (0, 1, 0) -> Audio only (Text & Visual missing)
    # (0, 0, 1) -> Visual only (Text & Audio missing)
    COMBOS = [
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1)
    ]

    COMBO_LABELS = {
        (1, 1, 1): ("T+A+V", "None"),
        (1, 1, 0): ("T+A",   "Visual"),
        (1, 0, 1): ("T+V",   "Audio"),
        (0, 1, 1): ("A+V",   "Text"),
        (1, 0, 0): ("T",     "Audio+Visual"),
        (0, 1, 0): ("A",     "Text+Visual"),
        (0, 0, 1): ("V",     "Text+Audio")
    }

    # Truly random selection: Uniform probability across all 7 combinations (p = 1/7 each)
    # NO PATTERN: Neither bandwidth constraint, nor text-first priority, nor skewed probabilities
    selected_indices = np.random.randint(0, len(COMBOS), size=total_samples)

    # Generate masks
    has_text = np.array([COMBOS[idx][0] for idx in selected_indices], dtype=int)
    has_audio = np.array([COMBOS[idx][1] for idx in selected_indices], dtype=int)
    has_visual = np.array([COMBOS[idx][2] for idx in selected_indices], dtype=int)
    combo_names = [COMBO_LABELS[COMBOS[idx]][0] for idx in selected_indices]
    missing_names = [COMBO_LABELS[COMBOS[idx]][1] for idx in selected_indices]

    # Save original text for reference/ground-truth
    df["Original_Text"] = df["Text"].astype(str)

    # Insert modality mask columns right after End_Time
    time_end_idx = df.columns.get_loc("End_Time")
    df.insert(time_end_idx + 1, "has_text", has_text)
    df.insert(time_end_idx + 2, "has_audio", has_audio)
    df.insert(time_end_idx + 3, "has_visual", has_visual)
    df.insert(time_end_idx + 4, "Modality_Combo", combo_names)
    df.insert(time_end_idx + 5, "Missing_Modalities", missing_names)

    # Mask text data when has_text == 0
    df["Text"] = np.where(df["has_text"] == 1, df["Original_Text"], mask_token)

    # Save new dataset to CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nNew dataset successfully saved to: {output_csv}")

    COMBO_TO_MISSING = {v[0]: v[1] for v in COMBO_LABELS.values()}

    # Print distribution statistics to verify pattern-free randomness
    print("\n" + "=" * 70)
    print("MODALITY COMBINATION DISTRIBUTION (Expected: ~14.29% each)")
    print("=" * 70)
    combo_dist = df["Modality_Combo"].value_counts()
    for combo, count in combo_dist.items():
        pct = (count / total_samples) * 100
        print(f"  {combo:<10} (Missing: {COMBO_TO_MISSING.get(combo, 'None'):<14}): {count:>5} samples ({pct:.2f}%)")

    print("\n" + "=" * 70)
    print("INDIVIDUAL MODALITY PRESENCE RATES (Expected: 4/7 = ~57.14% each)")
    print("=" * 70)
    print(f"  Text Present   : {df['has_text'].sum():>5} / {total_samples} ({df['has_text'].mean() * 100:.2f}%)")
    print(f"  Audio Present  : {df['has_audio'].sum():>5} / {total_samples} ({df['has_audio'].mean() * 100:.2f}%)")
    print(f"  Visual Present : {df['has_visual'].sum():>5} / {total_samples} ({df['has_visual'].mean() * 100:.2f}%)")

    print("\n" + "=" * 70)
    print("PREVIEW OF THE FIRST 10 SAMPLES IN NEW DATASET")
    print("=" * 70)
    preview_cols = ["Video_ID", "has_text", "has_audio", "has_visual", "Modality_Combo", "Missing_Modalities", "Text"]
    print(df[preview_cols].head(10).to_string(index=False))

    return df

if __name__ == "__main__":
    create_random_missing_dataset()
