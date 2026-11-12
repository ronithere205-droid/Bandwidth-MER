import os
import h5py
import numpy as np
import time

from bandwidth_simulation_engine import BandwidthCorruptor

DATA_DIR = "dataset/data"
OUTPUT_H5 = "mosei_bandwidth_simulated_sample.h5"

def run_simulation(max_utterances=100, delta_t=3.0, tau_audio=32.0, seed=42):
    print("=" * 80)
    print("CMU-MOSEI BANDWIDTH-AWARE MULTIMODAL SIMULATION PROTOCOL")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Data Extraction & Temporal Alignment (Direct CSD HDF5 access)
    # -------------------------------------------------------------------------
    label_file = os.path.join(DATA_DIR, "CMU_MOSEI_Labels.csd")
    covarep_file = os.path.join(DATA_DIR, "CMU_MOSEI_COVAREP.csd")
    words_file = os.path.join(DATA_DIR, "CMU_MOSEI_TimestampedWords.csd")
    glove_file = os.path.join(DATA_DIR, "CMU_MOSEI_TimestampedWordVectors.csd")

    print("[Step 1] Opening CSD files directly...")
    f_lab = h5py.File(label_file, "r")
    f_aud = h5py.File(covarep_file, "r")
    f_wrd = h5py.File(words_file, "r")
    f_glv = h5py.File(glove_file, "r")

    lab_data = f_lab["All Labels"]["data"]
    aud_data = f_aud["COVAREP"]["data"]
    wrd_data = f_wrd["words"]["data"]
    glv_data = f_glv["glove_vectors"]["data"]

    # -------------------------------------------------------------------------
    # STEPS 2-4: Window Segmentation, Dropout Simulation & Masking
    # -------------------------------------------------------------------------
    corruptor = BandwidthCorruptor(tau_audio=tau_audio, delta_t=delta_t, audio_fps=100.0, seed=seed)

    processed_samples = []
    total_audio_frames = 0
    total_dropped_frames = 0
    sample_for_timeline = None

    print(f"[Step 2-4] Simulating time-windowed bandwidth corruption (delta_t={delta_t}s, tau={tau_audio} kbps)...")
    
    for vid in lab_data.keys():
        if vid not in aud_data or vid not in wrd_data or vid not in glv_data:
            continue

        lab_ivs = lab_data[vid]["intervals"][:]
        lab_fts = lab_data[vid]["features"][:]
        aud_ivs = aud_data[vid]["intervals"][:]
        aud_fts = aud_data[vid]["features"][:]
        wrd_ivs = wrd_data[vid]["intervals"][:]
        wrd_fts = wrd_data[vid]["features"][:]
        glv_ivs = glv_data[vid]["intervals"][:]
        glv_fts = glv_data[vid]["features"][:]

        for u_idx, (start, end) in enumerate(lab_ivs):
            duration = end - start
            if duration < 1.0:  # Skip very short fragments (<1s)
                continue

            # Align Audio frames (100 Hz)
            s_idx = np.searchsorted(aud_ivs[:, 1], start, side="right")
            e_idx = np.searchsorted(aud_ivs[:, 0], end, side="left")
            if e_idx <= s_idx:
                continue

            clean_audio = aud_fts[s_idx:e_idx].astype(np.float32)
            clean_audio = np.nan_to_num(clean_audio, nan=0.0)

            # Align Words & GloVe
            w_mask = (wrd_ivs[:, 0] < end) & (wrd_ivs[:, 1] > start)
            words = []
            if np.any(w_mask):
                for w in wrd_fts[w_mask]:
                    t_str = w[0].decode("utf-8", errors="replace") if isinstance(w[0], bytes) else str(w[0])
                    if t_str.strip() and t_str.lower() != "sp":
                        words.append(t_str)
            text_str = " ".join(words)

            g_mask = (glv_ivs[:, 0] < end) & (glv_ivs[:, 1] > start)
            if np.any(g_mask):
                glove_seq = glv_fts[g_mask].astype(np.float32)
            else:
                glove_seq = np.zeros((1, 300), dtype=np.float32)

            # Target Emotion Labels
            labels = lab_fts[u_idx].astype(np.float32)

            # Apply Bandwidth Simulation & Masking (Steps 2, 3, 4)
            corrupt_audio, mask_audio, timeline = corruptor.corrupt_utterance(clean_audio, duration)

            n_frames = clean_audio.shape[0]
            n_dropped = int(np.sum(mask_audio == 0.0))
            total_audio_frames += n_frames
            total_dropped_frames += n_dropped

            sample_dict = {
                "vid": vid,
                "u_idx": u_idx,
                "start": start,
                "end": end,
                "duration": duration,
                "text": text_str,
                "glove": glove_seq,
                "clean_audio": clean_audio,
                "corrupt_audio": corrupt_audio,
                "mask_audio": mask_audio,
                "labels": labels,
                "timeline": timeline
            }
            processed_samples.append(sample_dict)

            # Pick a sample with multiple windows and at least one drop for detailed timeline display
            if sample_for_timeline is None and len(timeline) >= 3 and n_dropped > 0:
                sample_for_timeline = sample_dict

            if len(processed_samples) >= max_utterances:
                break

        if len(processed_samples) >= max_utterances:
            break

    f_lab.close()
    f_aud.close()
    f_wrd.close()
    f_glv.close()

    # -------------------------------------------------------------------------
    # STEP 5: Dataset Packaging & Verification
    # -------------------------------------------------------------------------
    print(f"\n[Step 5] Packaging {len(processed_samples)} processed utterances into HDF5 ({OUTPUT_H5})...")
    with h5py.File(OUTPUT_H5, "w") as out_h5:
        for i, s in enumerate(processed_samples):
            grp = out_h5.create_group(f"sample_{i:04d}")
            grp.attrs["video_id"] = s["vid"]
            grp.attrs["start_time"] = s["start"]
            grp.attrs["end_time"] = s["end"]
            grp.attrs["text"] = s["text"]
            
            grp.create_dataset("clean_audio", data=s["clean_audio"], compression="gzip")
            grp.create_dataset("corrupt_audio", data=s["corrupt_audio"], compression="gzip")
            grp.create_dataset("mask_audio", data=s["mask_audio"], compression="gzip")
            grp.create_dataset("text_glove", data=s["glove"], compression="gzip")
            grp.create_dataset("labels", data=s["labels"])

    overall_drop_pct = (total_dropped_frames / total_audio_frames) * 100.0 if total_audio_frames > 0 else 0
    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total Processed Utterances : {len(processed_samples)}")
    print(f"Total Audio Frames         : {total_audio_frames:,} (at 100 Hz)")
    print(f"Total Dropped Audio Frames : {total_dropped_frames:,}")
    print(f"Overall Audio Drop Rate    : {overall_drop_pct:.2f}%")
    print(f"Packaged HDF5 File Size    : {os.path.getsize(OUTPUT_H5) / (1024*1024):.2f} MB")

    # Display Single Sample Timeline Verification
    if sample_for_timeline:
        print("\n" + "=" * 80)
        print("SAMPLE UTTERANCE TIMELINE VERIFICATION (Step-by-Step Window Analysis)")
        print("=" * 80)
        s = sample_for_timeline
        print(f"Video ID     : {s['vid']}")
        print(f"Time Range   : {s['start']:.2f}s - {s['end']:.2f}s (Total Duration: {s['duration']:.2f}s)")
        print(f"Text Anchor  : \"{s['text']}\" [100% Intact]")
        print(f"Threshold    : tau_audio = {tau_audio} kbps")
        print("\nWindow Timeline:")
        print(f"{'Window':<8} {'Time Span':<16} {'Simulated B(t)':<16} {'Net State':<12} {'Status':<16} {'Mask (M_a)'}")
        print("-" * 80)
        
        for w in s["timeline"]:
            t_span = f"[{w['start_sec']:>4.1f}s - {w['end_sec']:>4.1f}s]"
            bw_str = f"{w['bandwidth_kbps']:>6.1f} kbps"
            if w["dropped"]:
                status_str = f"DROPOUT ({w['dropped_frames']} fr)"
                mask_str = f"PARTIAL ZERO (drop {w['dropped_frames']}/{w['window_frames']} frames)"
            else:
                status_str = "INTACT"
                mask_str = "ALL 1.0 (Full Pass)"
            print(f"{w['window']:<8} {t_span:<16} {bw_str:<16} {w['state']:<12} {status_str:<16} {mask_str}")
        print("=" * 80)

if __name__ == "__main__":
    run_simulation(max_utterances=50, delta_t=3.0, tau_audio=32.0, seed=42)

