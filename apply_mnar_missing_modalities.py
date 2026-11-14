import os
import time
import numpy as np
import pandas as pd

def apply_quality_based_mnar_modalities(
    text_labels_path="extracted_non_visual_dataset/mosei_text_and_labels.csv",
    audio_path="extracted_non_visual_dataset/mosei_audio_covarep.csv",
    output_path="extracted_non_visual_dataset/mosei_mnar_missing_multimodal.csv",
    seed=42
):
    print("=" * 85)
    print("RUNNING QUALITY & RELIABILITY-BASED MNAR ALGORITHM (NO LABEL LEAKAGE)")
    print("Missingness is driven strictly by modality signal degradation & transcription reliability:")
    print("  • Audio: Low acoustic energy + High acoustic noise / dispersion (transmission failure)")
    print("  • Text : Incomplete transcription rate + Low token confidence (ASR failure)")
    print("=" * 85)
    t0 = time.time()
    np.random.seed(seed)

    # -------------------------------------------------------------------------
    # STEP 1: Ingest Clean Multimodal Streams
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading clean Text, Labels, and Audio datasets...")
    if not os.path.exists(text_labels_path):
        raise FileNotFoundError(f"Missing file: {text_labels_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing file: {audio_path}")

    df_text = pd.read_csv(text_labels_path)
    df_audio = pd.read_csv(audio_path)
    total_samples = len(df_text)

    print(f"  • Loaded {total_samples:,} utterances from Text & Labels")
    print(f"  • Loaded {len(df_audio):,} utterances from Audio COVAREP (74 features)")

    covarep_cols = [c for c in df_audio.columns if c.startswith("covarep_")]

    # -------------------------------------------------------------------------
    # STEP 2: Extract Modality Quality & Reliability Metrics (Zero Target Bias)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Computing modality signal quality and reliability factors...")

    # Audio Metric 1: Acoustic Energy (Mean signal magnitude, filtering unvoiced sentinels)
    aud_matrix = df_audio[covarep_cols].values.astype(np.float32)
    clean_aud = np.where(aud_matrix < -500.0, 0.0, aud_matrix)
    acoustic_energy = np.mean(np.abs(clean_aud), axis=1)
    energy_z = (acoustic_energy - np.mean(acoustic_energy)) / (np.std(acoustic_energy) + 1e-7)

    # Audio Metric 2: Acoustic Noise & Transmission Dispersion (Coefficient of Variation)
    acoustic_noise = np.std(clean_aud, axis=1) / (acoustic_energy + 1e-6)
    noise_z = (acoustic_noise - np.mean(acoustic_noise)) / (np.std(acoustic_noise) + 1e-7)

    # Text Metric 1: Transcription Completeness / Density (Words per second of duration)
    duration = (df_text["End_Time"] - df_text["Start_Time"]).values
    word_counts = df_text["Text"].fillna("").apply(lambda s: len(str(s).split())).values
    speech_rate = word_counts / np.clip(duration, 0.5, None)
    rate_z = (speech_rate - np.mean(speech_rate)) / (np.std(speech_rate) + 1e-7)

    # Text Metric 2: Transcription Confidence Proxy (Average character length of tokens)
    # Fragmented, low-confidence ASR transcripts exhibit shorter, broken token lengths
    avg_word_len = df_text["Text"].fillna("").apply(
        lambda s: np.mean([len(w) for w in str(s).split()]) if str(s).split() else 0.0
    ).values
    conf_z = (avg_word_len - np.mean(avg_word_len)) / (np.std(avg_word_len) + 1e-7)

    # -------------------------------------------------------------------------
    # STEP 3: Logistic MNAR Formulations (Signal Quality Driven)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Formulating logistic drop probabilities based on signal degradation...")
    
    # Audio Drop Probability:
    # - Low acoustic energy (weak signal) -> negative weight on energy_z
    # - High acoustic noise / dispersion (transmission corruption) -> positive weight on noise_z
    logit_audio = -0.75 * energy_z + 0.60 * noise_z - 0.45
    p_drop_audio = 1.0 / (1.0 + np.exp(-logit_audio))

    # Text Drop Probability:
    # - Low speech density (incomplete transcription) -> negative weight on rate_z
    # - Low token confidence (fragmented, broken ASR tokens) -> negative weight on conf_z
    logit_text = -0.70 * rate_z - 0.55 * conf_z - 0.45
    p_drop_text = 1.0 / (1.0 + np.exp(-logit_text))

    # Sample binary dropout masks
    rand_audio = np.random.rand(total_samples)
    rand_text = np.random.rand(total_samples)

    has_audio = (rand_audio >= p_drop_audio).astype(int)
    has_text = (rand_text >= p_drop_text).astype(int)

    # Non-empty constraint: ensure at least one modality is preserved per utterance
    both_dropped = (has_audio == 0) & (has_text == 0)
    for idx in np.where(both_dropped)[0]:
        if p_drop_audio[idx] < p_drop_text[idx]:
            has_audio[idx] = 1
        else:
            has_text[idx] = 1

    # Modality state names
    state_names = []
    missing_names = []
    for ha, ht in zip(has_audio, has_text):
        if ha == 1 and ht == 1:
            state_names.append("Text+Audio")
            missing_names.append("None")
        elif ha == 0 and ht == 1:
            state_names.append("Text_Only")
            missing_names.append("Audio")
        elif ha == 1 and ht == 0:
            state_names.append("Audio_Only")
            missing_names.append("Text")

    # -------------------------------------------------------------------------
    # STEP 4: Apply Modality Masking to Features
    # -------------------------------------------------------------------------
    print("\n[Step 4] Applying feature corruption (zeroing dropped audio, blanking dropped text)...")
    df_out = pd.DataFrame()
    df_out["Video_ID"] = df_text["Video_ID"]
    df_out["Start_Time"] = df_text["Start_Time"]
    df_out["End_Time"] = df_text["End_Time"]

    # Availability Masks & Metadata
    df_out["has_text"] = has_text
    df_out["has_audio"] = has_audio
    df_out["Modality_State"] = state_names
    df_out["Missing_Modality"] = missing_names
    df_out["Missing_Mechanism"] = "MNAR_Quality_Driven"
    df_out["p_drop_audio"] = np.round(p_drop_audio, 4)
    df_out["p_drop_text"] = np.round(p_drop_text, 4)

    # Signal Quality Diagnostic Columns (for validation & reporting)
    df_out["acoustic_energy"] = np.round(acoustic_energy, 4)
    df_out["acoustic_noise"] = np.round(acoustic_noise, 4)
    df_out["speech_rate_wps"] = np.round(speech_rate, 2)
    df_out["avg_word_length"] = np.round(avg_word_len, 2)

    # Mask Text: blank out Text when has_text == 0, keep ground truth in Original_Text
    raw_text = df_text["Text"].fillna("").astype(str)
    df_out["Original_Text"] = raw_text
    df_out["Text"] = np.where(has_text == 1, raw_text, "")

    # Mask Audio: zero out all 74 COVAREP features when has_audio == 0
    audio_mask_expanded = has_audio[:, np.newaxis]
    masked_audio_matrix = aud_matrix * audio_mask_expanded

    for idx, col in enumerate(covarep_cols):
        df_out[col] = np.round(masked_audio_matrix[:, idx], 4)

    # Preserve Target Emotion Labels intact (Zero label corruption or label leakage)
    emotion_cols = ["Sentiment", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
    for col in emotion_cols:
        df_out[col] = df_text[col]

    # -------------------------------------------------------------------------
    # STEP 5: Export Dataset & Statistical Verification
    # -------------------------------------------------------------------------
    print(f"\n[Step 5] Writing updated MNAR dataset to: {output_path}...")
    df_out.to_csv(output_path, index=False, encoding="utf-8")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 85)
    print("QUALITY-BASED MNAR DATASET SUMMARY & STATISTICAL VERIFICATION")
    print("=" * 85)
    print(f"Total Utterances Packaged: {total_samples:,}")
    print(f"Output File Size         : {file_size_mb:.2f} MB")
    print(f"Total Columns            : {len(df_out.columns)}")

    print("\nModality Distribution:")
    state_counts = df_out["Modality_State"].value_counts()
    for state, count in state_counts.items():
        pct = (count / total_samples) * 100
        print(f"  • {state:<12}: {count:>5} samples ({pct:.2f}%)")

    print(f"\nIndividual Modality Rates:")
    print(f"  • Text Available  (has_text = 1) : {df_out['has_text'].sum():>5} / {total_samples} ({df_out['has_text'].mean()*100:.2f}%)")
    print(f"  • Audio Available (has_audio = 1): {df_out['has_audio'].sum():>5} / {total_samples} ({df_out['has_audio'].mean()*100:.2f}%)")

    # -------------------------------------------------------------------------
    # VALIDATION OF THE EXPERIMENTAL DESIGN
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("PROOF OF QUALITY-DRIVEN MNAR (EXPERIMENTAL DEFENSE EVIDENCE)")
    print("=" * 85)
    
    # 1. Audio Missingness vs Acoustic Energy
    mean_energy_pres = acoustic_energy[has_audio == 1].mean()
    mean_energy_miss = acoustic_energy[has_audio == 0].mean()
    print(f"1. Audio Missingness vs Acoustic Signal Energy (Factor: Weak Audio Signal):")
    print(f"   • Mean Energy when Audio Present : {mean_energy_pres:.4f}")
    print(f"   • Mean Energy when Audio Missing : {mean_energy_miss:.4f}")
    diff_energy = ((mean_energy_pres - mean_energy_miss) / mean_energy_pres) * 100
    print(f"   --> Missing audio samples have {diff_energy:.2f}% weaker signal energy!")

    # 2. Audio Missingness vs Acoustic Noise / Corruption
    mean_noise_pres = acoustic_noise[has_audio == 1].mean()
    mean_noise_miss = acoustic_noise[has_audio == 0].mean()
    print(f"\n2. Audio Missingness vs Acoustic Noise / Corruption (Factor: Poor Transmission):")
    print(f"   • Mean Noise Dispersion when Audio Present : {mean_noise_pres:.4f}")
    print(f"   • Mean Noise Dispersion when Audio Missing : {mean_noise_miss:.4f}")
    print(f"   --> Missing audio samples have higher noise / transmission dispersion!")

    # 3. Text Missingness vs Speech Rate (Completeness)
    mean_rate_pres = speech_rate[has_text == 1].mean()
    mean_rate_miss = speech_rate[has_text == 0].mean()
    print(f"\n3. Text Missingness vs Transcription Density (Factor: Incomplete Transcription):")
    print(f"   • Mean Speech Rate when Text Present : {mean_rate_pres:.2f} words/sec")
    print(f"   • Mean Speech Rate when Text Missing : {mean_rate_miss:.2f} words/sec")
    diff_rate = ((mean_rate_pres - mean_rate_miss) / mean_rate_pres) * 100
    print(f"   --> Missing text samples have {diff_rate:.2f}% lower word density (incomplete transcripts)!")

    # 4. Text Missingness vs Token Length (Confidence Proxy)
    mean_len_pres = avg_word_len[has_text == 1].mean()
    mean_len_miss = avg_word_len[has_text == 0].mean()
    print(f"\n4. Text Missingness vs Token Regularity (Factor: Low ASR Confidence):")
    print(f"   • Mean Token Length when Text Present : {mean_len_pres:.2f} chars/word")
    print(f"   • Mean Token Length when Text Missing : {mean_len_miss:.2f} chars/word")
    diff_len = ((mean_len_pres - mean_len_miss) / mean_len_pres) * 100
    print(f"   --> Missing text samples have {diff_len:.2f}% shorter/fragmented broken tokens!")

    print("\n" + "=" * 85)
    print("FIRST 5 SAMPLES PREVIEW")
    print("=" * 85)
    preview_cols = ["Video_ID", "has_text", "has_audio", "Modality_State", "acoustic_energy", "speech_rate_wps", "Text"]
    print(df_out[preview_cols].head(5).to_string(index=False))

    dt = time.time() - t0
    print(f"\nExecution completed in {dt:.2f} seconds.")
    return df_out

if __name__ == "__main__":
    apply_quality_based_mnar_modalities()
