import h5py
import numpy as np

H5_PATH = "mosei_bandwidth_simulated_sample.h5"

def inspect_h5_file(file_path=H5_PATH):
    print("=" * 80)
    print(f"INSPECTING HDF5 DATASET: {file_path}")
    print("=" * 80)

    with h5py.File(file_path, "r") as f:
        sample_keys = list(f.keys())
        total_samples = len(sample_keys)
        print(f"Total Utterances Packaged: {total_samples}")
        print(f"Sample Groups Available  : {sample_keys[:5]} ... to {sample_keys[-1]}\n")

        # ---------------------------------------------------------------------
        # 1. Structure of a single sample
        # ---------------------------------------------------------------------
        first_key = sample_keys[0]
        sample = f[first_key]
        
        print(f"--- Group: '{first_key}' Structure ---")
        print("Attributes (Metadata):")
        for attr_name, attr_val in sample.attrs.items():
            print(f"  • {attr_name:<12} : {attr_val}")

        print("\nStored Datasets (Tensors):")
        for ds_name in sample.keys():
            ds = sample[ds_name]
            print(f"  • {ds_name:<14} : Shape {str(ds.shape):<14} | Dtype: {ds.dtype}")

        # ---------------------------------------------------------------------
        # 2. Inspecting an utterance that experienced an audio dropout
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("INSPECTING A DROPOUT SAMPLE (Comparing Clean vs Corrupt vs Mask)")
        print("=" * 80)

        found_drop = False
        for k in sample_keys:
            s = f[k]
            mask = s["mask_audio"][:]
            if np.sum(mask == 0.0) > 0:
                found_drop = True
                clean_audio = s["clean_audio"][:]
                corrupt_audio = s["corrupt_audio"][:]
                glove = s["text_glove"][:]
                labels = s["labels"][:]

                dropped_frames = int(np.sum(mask == 0.0))
                total_frames = len(mask)
                drop_ratio = (dropped_frames / total_frames) * 100

                print(f"Sample ID        : {k}")
                print(f"Video ID         : {s.attrs['video_id']}")
                print(f"Time Range       : {s.attrs['start_time']:.2f}s - {s.attrs['end_time']:.2f}s")
                print(f"Transcript (Text): \"{s.attrs['text']}\"")
                print(f"Total Frames     : {total_frames} (at 100 Hz = {total_frames*0.01:.2f}s duration)")
                print(f"Dropped Frames   : {dropped_frames} frames ({drop_ratio:.1f}% dropped)")
                print(f"GloVe Sequence   : Shape {glove.shape} ({glove.shape[0]} words x 300 dim)")
                print(f"Emotion Targets  : {labels.round(2)} (Sentiment, Happy, Sad, Angry, Fear, Disgust, Surprise)\n")

                # Locate where mask is 0
                drop_indices = np.where(mask == 0.0)[0]
                first_drop_idx = drop_indices[0]
                
                print(f"Frame-by-Frame Comparison at Dropout Index {first_drop_idx}:")
                print(f"  Mask value             : M_a[{first_drop_idx}] = {mask[first_drop_idx]}")
                print(f"  Clean Audio (first 5)  : {clean_audio[first_drop_idx][:5]}")
                print(f"  Corrupt Audio (first 5): {corrupt_audio[first_drop_idx][:5]}  <-- Zeroed out by mask!")
                print(f"  Clean Audio Norm       : {np.linalg.norm(clean_audio[first_drop_idx]):.4f}")
                print(f"  Corrupt Audio Norm     : {np.linalg.norm(corrupt_audio[first_drop_idx]):.4f}")
                break

        if not found_drop:
            print("No dropout samples found in this batch.")

        print("=" * 80)

if __name__ == "__main__":
    inspect_h5_file()

