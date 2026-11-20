import os
import pandas as pd
import numpy as np

def create_complete_dataset(
    text_labels_path="extracted_non_visual_dataset/mosei_text_and_labels.csv",
    audio_path="extracted_non_visual_dataset/mosei_audio_covarep.csv",
    output_path="extracted_non_visual_dataset/mosei_complete_multimodal.csv"
):
    print("=" * 80)
    print("GENERATING TRUE COMPLETE MULTIMODAL DATASET (CLEAN UPPER BOUND BASELINE)")
    print("=" * 80)

    df_text = pd.read_csv(text_labels_path)
    df_audio = pd.read_csv(audio_path)
    total_samples = len(df_text)

    covarep_cols = [c for c in df_audio.columns if c.startswith("covarep_")]

    df_out = pd.DataFrame()
    df_out["Video_ID"] = df_text["Video_ID"]
    df_out["Start_Time"] = df_text["Start_Time"]
    df_out["End_Time"] = df_text["End_Time"]

    # In Complete Dataset, all modalities are present (100% available)
    has_text = (df_text["Text"].fillna("").str.strip() != "").astype(int)
    has_audio = df_audio["audio_detected"].fillna(0).astype(int)

    df_out["has_text"] = has_text
    df_out["has_audio"] = has_audio
    df_out["Modality_State"] = "Text+Audio"
    df_out["Missing_Modality"] = "None"
    df_out["Missing_Mechanism"] = "Complete_Clean"

    # Clean text (100% preserved)
    raw_text = df_text["Text"].fillna("").astype(str)
    df_out["Text"] = raw_text
    df_out["Original_Text"] = raw_text

    # Clean audio (100% unmasked true COVAREP features)
    for col in covarep_cols:
        df_out[col] = df_audio[col]

    # Target labels
    emotion_cols = ["Sentiment", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
    for col in emotion_cols:
        df_out[col] = df_text[col]

    df_out.to_csv(output_path, index=False, encoding="utf-8")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nSaved True Complete Dataset to: {output_path}")
    print(f"Total Utterances: {total_samples:,}")
    print(f"File Size       : {file_size_mb:.2f} MB")
    print(f"Text Available  : {(has_text == 1).sum()} / {total_samples} ({(has_text == 1).mean()*100:.2f}%)")
    print(f"Audio Available : {(has_audio == 1).sum()} / {total_samples} ({(has_audio == 1).mean()*100:.2f}%)")
    print("=" * 80)
    return df_out

if __name__ == "__main__":
    create_complete_dataset()
