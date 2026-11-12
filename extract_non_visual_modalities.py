import os
import time
import h5py
import numpy as np
import pandas as pd

def extract_non_visual_modalities(
    data_dir="dataset/data",
    output_dir="extracted_non_visual_dataset",
    seed=42
):
    print("=" * 80)
    print("CMU-MOSEI NON-VISUAL MODALITY EXTRACTION SCRIPT")
    print("Extracting: Labels, Words (Text), Word Vectors (GloVe), Phones, Audio (COVAREP)")
    print("Excluding : VisualFacet42.csd, OpenFace2.csd (Visuals Excluded)")
    print("=" * 80)

    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    # File paths for the 5 non-visual computational sequences
    file_paths = {
        "labels": os.path.join(data_dir, "CMU_MOSEI_Labels.csd"),
        "words": os.path.join(data_dir, "CMU_MOSEI_TimestampedWords.csd"),
        "phones": os.path.join(data_dir, "CMU_MOSEI_TimestampedPhones.csd"),
        "audio": os.path.join(data_dir, "CMU_MOSEI_COVAREP.csd"),
        "glove": os.path.join(data_dir, "CMU_MOSEI_TimestampedWordVectors.csd")
    }

    # Verify input files exist
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset file for {name}: {path}")

    emotion_names = ["Sentiment", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
    covarep_cols = [f"covarep_{i}" for i in range(74)]

    # Storage containers for extracted records
    text_label_rows = []
    audio_rows = []
    phone_rows = []
    glove_vectors = []
    multimodal_rows = []

    print("\n[1/4] Opening HDF5 computational sequence files...")
    f_lab = h5py.File(file_paths["labels"], "r")
    f_wrd = h5py.File(file_paths["words"], "r")
    f_phn = h5py.File(file_paths["phones"], "r")
    f_aud = h5py.File(file_paths["audio"], "r")
    f_glv = h5py.File(file_paths["glove"], "r")

    lab_dict = f_lab["All Labels"]["data"]
    wrd_dict = f_wrd["words"]["data"]
    phn_dict = f_phn["phoneme"]["data"]
    aud_dict = f_aud["COVAREP"]["data"]
    glv_dict = f_glv["glove_vectors"]["data"]

    video_ids = list(lab_dict.keys())
    total_videos = len(video_ids)
    print(f"Total videos with labels: {total_videos}")

    print("\n[2/4] Processing utterances and aligning modalities across time intervals...")
    processed_utterances = 0

    for v_idx, vid in enumerate(video_ids):
        # Progress reporting
        if (v_idx + 1) % 500 == 0 or (v_idx + 1) == total_videos:
            elapsed = time.time() - t_start
            print(f"  Processed {v_idx + 1}/{total_videos} videos ({processed_utterances} utterances) - Elapsed: {elapsed:.1f}s")

        # Label intervals and targets
        lab_intervals = lab_dict[vid]["intervals"][:]
        lab_features = lab_dict[vid]["features"][:]

        # Pre-fetch video data if present in sequences
        has_wrd = vid in wrd_dict
        has_phn = vid in phn_dict
        has_aud = vid in aud_dict
        has_glv = vid in glv_dict

        if has_wrd:
            wrd_ivs = wrd_dict[vid]["intervals"][:]
            wrd_fts = wrd_dict[vid]["features"][:]

        if has_phn:
            phn_ivs = phn_dict[vid]["intervals"][:]
            phn_fts = phn_dict[vid]["features"][:]

        if has_aud:
            aud_ivs = aud_dict[vid]["intervals"][:]
            aud_fts = aud_dict[vid]["features"][:]

        if has_glv:
            glv_ivs = glv_dict[vid]["intervals"][:]
            glv_fts = glv_dict[vid]["features"][:]

        # Process each utterance interval defined by ground-truth labels
        for u_idx, (start, end) in enumerate(lab_intervals):
            processed_utterances += 1
            lbl_vals = lab_features[u_idx]

            # ----------------------------------------------------
            # 1. Words / Text Extraction
            # ----------------------------------------------------
            sentence = ""
            if has_wrd and len(wrd_ivs) > 0:
                w_mask = (wrd_ivs[:, 0] < end) & (wrd_ivs[:, 1] > start)
                if np.any(w_mask):
                    words = []
                    for w in wrd_fts[w_mask]:
                        token = w[0].decode("utf-8", errors="replace") if isinstance(w[0], bytes) else str(w[0])
                        token = token.strip()
                        if token and token.lower() != "sp":
                            words.append(token)
                    sentence = " ".join(words)

            # ----------------------------------------------------
            # 2. Phonemes Extraction
            # ----------------------------------------------------
            phonemes_str = ""
            if has_phn and len(phn_ivs) > 0:
                p_mask = (phn_ivs[:, 0] < end) & (phn_ivs[:, 1] > start)
                if np.any(p_mask):
                    phones = []
                    for p in phn_fts[p_mask]:
                        phone_token = p[0].decode("utf-8", errors="replace") if isinstance(p[0], bytes) else str(p[0])
                        phone_token = phone_token.strip()
                        if phone_token and phone_token.lower() != "sp":
                            phones.append(phone_token)
                    phonemes_str = " ".join(phones)

            # ----------------------------------------------------
            # 3. Audio (COVAREP - 74 Acoustic Features) Extraction
            # ----------------------------------------------------
            covarep_vec = np.zeros(74, dtype=np.float32)
            audio_found = False
            if has_aud and len(aud_ivs) > 0:
                s_idx = np.searchsorted(aud_ivs[:, 1], start, side="right")
                e_idx = np.searchsorted(aud_ivs[:, 0], end, side="left")
                if e_idx > s_idx:
                    covarep_vec = np.mean(aud_fts[s_idx:e_idx], axis=0).astype(np.float32)
                    # Replace any NaNs with 0
                    covarep_vec = np.nan_to_num(covarep_vec, nan=0.0)
                    audio_found = True

            # ----------------------------------------------------
            # 4. GloVe Word Vectors (300-d) Extraction
            # ----------------------------------------------------
            glove_vec = np.zeros(300, dtype=np.float32)
            if has_glv and len(glv_ivs) > 0:
                g_mask = (glv_ivs[:, 0] < end) & (glv_ivs[:, 1] > start)
                if np.any(g_mask):
                    glove_vec = np.mean(glv_fts[g_mask], axis=0).astype(np.float32)
                    glove_vec = np.nan_to_num(glove_vec, nan=0.0)

            # ----------------------------------------------------
            # 5. Bandwidth Awareness Simulation (Option B)
            # ----------------------------------------------------
            # Costs: Text = 1 kbps, Audio = 32 kbps
            # Simulate network channels: Low (10-25 kbps), High (40-100 kbps), Lossy
            rand_val = np.random.rand()
            if rand_val < 0.35:
                # Low Bandwidth (2G / Edge): Budget < 32 kbps -> Audio dropped, Text preserved
                sim_budget = np.random.uniform(5.0, 25.0)
                has_t, has_a = 1, 0
                net_tier = "Low_Bandwidth_2G"
                req_cost = 1.0
            elif rand_val < 0.85:
                # High Bandwidth (3G / 4G / WiFi): Budget >= 32 kbps -> Both preserved
                sim_budget = np.random.uniform(40.0, 150.0)
                has_t, has_a = 1, 1
                net_tier = "High_Bandwidth_3G_4G"
                req_cost = 33.0
            else:
                # Packet loss / Sensor dropout
                sim_budget = np.random.uniform(30.0, 80.0)
                net_tier = "Lossy_Channel"
                if np.random.rand() < 0.5:
                    has_t, has_a = 1, 0  # Audio dropped due to packet loss
                    req_cost = 1.0
                else:
                    has_t, has_a = 0, 1  # Text dropped due to packet loss
                    req_cost = 32.0

            # Store individual modality rows
            base_info = {
                "Video_ID": vid,
                "Start_Time": round(float(start), 3),
                "End_Time": round(float(end), 3),
            }

            # Text and Labels row
            t_row = dict(base_info)
            t_row["Text"] = sentence
            for k, emo in enumerate(emotion_names):
                t_row[emo] = round(float(lbl_vals[k]), 4)
            text_label_rows.append(t_row)

            # Audio row
            a_row = dict(base_info)
            a_row["audio_detected"] = 1 if audio_found else 0
            for k, c_name in enumerate(covarep_cols):
                a_row[c_name] = round(float(covarep_vec[k]), 4)
            audio_rows.append(a_row)

            # Phonemes row
            p_row = dict(base_info)
            p_row["Phonemes"] = phonemes_str
            phone_rows.append(p_row)

            # GloVe vector storage
            glove_vectors.append(glove_vec)

            # Unified Multimodal Non-Visual row
            m_row = dict(base_info)
            m_row["Text"] = sentence if has_t == 1 else ""
            m_row["Original_Text"] = sentence
            m_row["Phonemes"] = phonemes_str
            m_row["has_text"] = has_t
            m_row["has_audio"] = has_a
            m_row["Bandwidth_Budget_kbps"] = round(sim_budget, 1)
            m_row["Required_Cost_kbps"] = req_cost
            m_row["Network_Tier"] = net_tier
            for k, c_name in enumerate(covarep_cols):
                # When has_audio == 0, acoustic features are masked (0.0)
                val = covarep_vec[k] if has_a == 1 else 0.0
                m_row[c_name] = round(float(val), 4)
            for k, emo in enumerate(emotion_names):
                m_row[emo] = round(float(lbl_vals[k]), 4)
            multimodal_rows.append(m_row)

    # Close open HDF5 file handles
    f_lab.close()
    f_wrd.close()
    f_phn.close()
    f_aud.close()
    f_glv.close()

    print(f"\n[3/4] Extracted all {processed_utterances} utterances across all 5 non-visual sequences.")

    # ----------------------------------------------------
    # Save Files into the New Output Directory
    # ----------------------------------------------------
    print("\n[4/4] Writing extracted files into target folder:", output_dir)

    # 1. Text & Labels CSV
    f1 = os.path.join(output_dir, "mosei_text_and_labels.csv")
    df_text = pd.DataFrame(text_label_rows)
    df_text.to_csv(f1, index=False, encoding="utf-8")
    print(f"  -> Saved: {f1} ({os.path.getsize(f1) / (1024*1024):.2f} MB)")

    # 2. Audio COVAREP CSV (74 features)
    f2 = os.path.join(output_dir, "mosei_audio_covarep.csv")
    df_audio = pd.DataFrame(audio_rows)
    df_audio.to_csv(f2, index=False, encoding="utf-8")
    print(f"  -> Saved: {f2} ({os.path.getsize(f2) / (1024*1024):.2f} MB)")

    # 3. Phonemes CSV
    f3 = os.path.join(output_dir, "mosei_phonemes.csv")
    df_phone = pd.DataFrame(phone_rows)
    df_phone.to_csv(f3, index=False, encoding="utf-8")
    print(f"  -> Saved: {f3} ({os.path.getsize(f3) / (1024*1024):.2f} MB)")

    # 4. GloVe Word Vectors (.npz)
    f4 = os.path.join(output_dir, "mosei_glove_300d_vectors.npz")
    glove_arr = np.array(glove_vectors, dtype=np.float32)
    np.savez_compressed(
        f4,
        features=glove_arr,
        video_ids=np.array(df_text["Video_ID"].values, dtype=str),
        start_times=df_text["Start_Time"].values,
        end_times=df_text["End_Time"].values
    )
    print(f"  -> Saved: {f4} ({os.path.getsize(f4) / (1024*1024):.2f} MB)")

    # 5. Master Multimodal Non-Visual Dataset (with Option B Bandwidth Awareness)
    f5 = os.path.join(output_dir, "mosei_multimodal_non_visual.csv")
    df_multi = pd.DataFrame(multimodal_rows)
    df_multi.to_csv(f5, index=False, encoding="utf-8")
    print(f"  -> Saved: {f5} ({os.path.getsize(f5) / (1024*1024):.2f} MB)")

    # 6. Extraction Summary Documentation
    summary_path = os.path.join(output_dir, "README_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as s_file:
        s_file.write("CMU-MOSEI NON-VISUAL MODALITY EXTRACTION SUMMARY\n")
        s_file.write("=" * 60 + "\n\n")
        s_file.write(f"Total Utterances Extracted : {processed_utterances}\n")
        s_file.write(f"Total Videos Processed    : {total_videos}\n")
        s_file.write(f"Visual Sequences Excluded : CMU_MOSEI_VisualFacet42.csd, CMU_MOSEI_OpenFace2.csd\n\n")
        s_file.write("FILES IN THIS DIRECTORY:\n")
        s_file.write(f"1. mosei_text_and_labels.csv    : Video_ID, Start_Time, End_Time, Text, 7 Emotion/Sentiment Targets\n")
        s_file.write(f"2. mosei_audio_covarep.csv      : Video_ID, Start_Time, End_Time, audio_detected, 74 COVAREP features\n")
        s_file.write(f"3. mosei_phonemes.csv           : Video_ID, Start_Time, End_Time, Phonemes sequence\n")
        s_file.write(f"4. mosei_glove_300d_vectors.npz : Compressed numpy array of 300-d GloVe embeddings (shape: {glove_arr.shape})\n")
        s_file.write(f"5. mosei_multimodal_non_visual.csv: Unified master dataset combining Text, Phonemes, 74 COVAREP\n")
        s_file.write(f"                                   audio features, labels, and Option B Bandwidth-Awareness columns\n")
        s_file.write(f"                                   (Bandwidth_Budget_kbps, Required_Cost_kbps, Network_Tier, has_text, has_audio)\n")
    print(f"  -> Saved: {summary_path}")

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL NON-VISUAL MODALITIES EXTRACTED SUCCESSFULLY IN {total_time:.1f}s ({total_time/60:.2f} min)!")
    print(f"All files saved into folder: {os.path.abspath(output_dir)}")
    print("=" * 80)

if __name__ == "__main__":
    extract_non_visual_modalities()

