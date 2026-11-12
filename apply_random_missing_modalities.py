import os
import time
import numpy as np
import pandas as pd

def apply_random_missing_modalities(
    text_labels_path="extracted_non_visual_dataset/mosei_text_and_labels.csv",
    audio_path="extracted_non_visual_dataset/mosei_audio_covarep.csv",
    output_path="extracted_non_visual_dataset/mosei_random_missing_multimodal.csv",
    mode="uniform_combinations",  # "uniform_combinations" or "independent_bernoulli"
    bernoulli_p=0.4,              # used only if mode == "independent_bernoulli"
    seed=42
):
    print("=" * 80)
    print("RUNNING RANDOM MISSING MODALITY ALGORITHM (MCAR - NO PATTERN)")
    print("=" * 80)
    t0 = time.time()

    np.random.seed(seed)

    # -------------------------------------------------------------------------
    # STEP 1: Load Clean Multimodal Modalities
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading clean Text, Labels, and Audio datasets...")
    if not os.path.exists(text_labels_path):
        raise FileNotFoundError(f"Missing: {text_labels_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing: {audio_path}")

    df_text = pd.read_csv(text_labels_path)
    df_audio = pd.read_csv(audio_path)

    total_samples = len(df_text)
    print(f"  • Loaded {total_samples:,} utterances from Text & Labels")
    print(f"  • Loaded {len(df_audio):,} utterances from Audio COVAREP (74 features)")

    # -------------------------------------------------------------------------
    # STEP 2: Formulate Random Missingness (MCAR)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Sampling random modality states without any deterministic pattern...")
    
    # Valid non-empty combinations: (has_text, has_audio)
    # (1, 1) -> Both present (Complete)
    # (1, 0) -> Text only (Audio missing)
    # (0, 1) -> Audio only (Text missing)
    COMBINATIONS = [
        (1, 1),  # Text + Audio
        (1, 0),  # Text Only
        (0, 1)   # Audio Only
    ]

    COMBO_LABELS = {
        (1, 1): ("Text+Audio", "None"),
        (1, 0): ("Text_Only",  "Audio"),
        (0, 1): ("Audio_Only", "Text")
    }

    if mode == "uniform_combinations":
        # Each valid combination has exactly 1/3 (33.33%) probability
        sampled_indices = np.random.randint(0, len(COMBINATIONS), size=total_samples)
        has_text_arr = np.array([COMBINATIONS[i][0] for i in sampled_indices], dtype=int)
        has_audio_arr = np.array([COMBINATIONS[i][1] for i in sampled_indices], dtype=int)
    elif mode == "independent_bernoulli":
        # Each modality independently dropped with probability bernoulli_p
        has_text_arr = (np.random.rand(total_samples) > bernoulli_p).astype(int)
        has_audio_arr = (np.random.rand(total_samples) > bernoulli_p).astype(int)
        # Ensure at least one modality remains (no empty samples)
        both_zero = (has_text_arr == 0) & (has_audio_arr == 0)
        has_text_arr[both_zero] = 1

    state_names = [COMBO_LABELS[(has_text_arr[i], has_audio_arr[i])][0] for i in range(total_samples)]
    missing_names = [COMBO_LABELS[(has_text_arr[i], has_audio_arr[i])][1] for i in range(total_samples)]

    # -------------------------------------------------------------------------
    # STEP 3: Apply Modality Masking to Features
    # -------------------------------------------------------------------------
    print("\n[Step 3] Applying modality masking (zeroing dropped features, blanking dropped text)...")

    # Audio feature column names (covarep_0 to covarep_73)
    covarep_cols = [c for c in df_audio.columns if c.startswith("covarep_")]

    # Create merged dataframe starting with identifiers
    df_out = pd.DataFrame()
    df_out["Video_ID"] = df_text["Video_ID"]
    df_out["Start_Time"] = df_text["Start_Time"]
    df_out["End_Time"] = df_text["End_Time"]

    # Availability Masks & State Metadata
    df_out["has_text"] = has_text_arr
    df_out["has_audio"] = has_audio_arr
    df_out["Modality_State"] = state_names
    df_out["Missing_Modality"] = missing_names

    # Mask Text: preserve ground truth in Original_Text, blank Text when has_text == 0
    raw_text = df_text["Text"].fillna("").astype(str)
    df_out["Original_Text"] = raw_text
    df_out["Text"] = np.where(has_text_arr == 1, raw_text, "")

    # Mask Audio: zero out all 74 covarep features when has_audio == 0
    audio_matrix = df_audio[covarep_cols].values.astype(np.float32)
    # Broadcast mask (N, 1) across (N, 74) features
    audio_mask_expanded = has_audio_arr[:, np.newaxis]
    masked_audio_matrix = audio_matrix * audio_mask_expanded

    for idx, col in enumerate(covarep_cols):
        df_out[col] = np.round(masked_audio_matrix[:, idx], 4)

    # -------------------------------------------------------------------------
    # STEP 4: Preserve Target Emotion Labels
    # -------------------------------------------------------------------------
    print("\n[Step 4] Preserving ground-truth emotion and sentiment labels (100% untouched)...")
    emotion_cols = ["Sentiment", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
    for col in emotion_cols:
        df_out[col] = df_text[col]

    # -------------------------------------------------------------------------
    # STEP 5: Export Dataset & Compute Summary Statistics
    # -------------------------------------------------------------------------
    print(f"\n[Step 5] Writing output dataset to: {output_path}...")
    df_out.to_csv(output_path, index=False, encoding="utf-8")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 80)
    print("SIMULATION RESULTS & DISTRIBUTION STATISTICS (MCAR VERIFICATION)")
    print("=" * 80)
    print(f"Total Utterances Packaged: {total_samples:,}")
    print(f"Output File Size         : {file_size_mb:.2f} MB")
    print(f"Total Columns            : {len(df_out.columns)} (Identifiers + Masks + Text + 74 Audio + Labels)")

    print("\nModality Combination Distribution (Expected: ~33.33% each):")
    state_counts = df_out["Modality_State"].value_counts()
    for state, count in state_counts.items():
        pct = (count / total_samples) * 100
        print(f"  • {state:<12} (Missing: {COMBO_LABELS[[k for k,v in COMBO_LABELS.items() if v[0]==state][0]][1]:<6}): {count:>5} samples ({pct:.2f}%)")

    print("\nIndividual Modality Availability Rates (Expected: ~66.67% each):")
    text_pct = (df_out["has_text"].sum() / total_samples) * 100
    audio_pct = (df_out["has_audio"].sum() / total_samples) * 100
    print(f"  • Text Available  (has_text = 1) : {df_out['has_text'].sum():>5} / {total_samples} ({text_pct:.2f}%)")
    print(f"  • Audio Available (has_audio = 1): {df_out['has_audio'].sum():>5} / {total_samples} ({audio_pct:.2f}%)")

    print("\n" + "=" * 80)
    print("FIRST 5 SAMPLES PREVIEW")
    print("=" * 80)
    preview_cols = ["Video_ID", "has_text", "has_audio", "Modality_State", "Missing_Modality", "Text"]
    print(df_out[preview_cols].head(5).to_string(index=False))

    dt = time.time() - t0
    print(f"\nExecution completed in {dt:.2f} seconds.")
    return df_out

if __name__ == "__main__":
    apply_random_missing_modalities()

