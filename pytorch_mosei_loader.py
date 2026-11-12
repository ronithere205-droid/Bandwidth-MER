import h5py
import numpy as np

# We provide a standalone Dataset class that works with or without torch installed
class BandwidthMoseiDataset:
    """
    Dataset loader for mosei_bandwidth_simulated_sample.h5
    Supports random access, indexing, and returns numpy/torch tensors.
    """
    def __init__(self, h5_path="mosei_bandwidth_simulated_sample.h5"):
        self.h5_path = h5_path
        # Cache sample keys
        with h5py.File(self.h5_path, "r") as f:
            self.keys = list(f.keys())
        self.f = None  # Lazy file handle for multi-process safety

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.h5_path, "r")

        key = self.keys[idx]
        sample = self.f[key]

        data = {
            "sample_id": key,
            "video_id": sample.attrs["video_id"],
            "start_time": sample.attrs["start_time"],
            "end_time": sample.attrs["end_time"],
            "text": sample.attrs["text"],
            "text_glove": sample["text_glove"][:],          # Shape: (N_words, 300)
            "clean_audio": sample["clean_audio"][:],        # Shape: (L_audio, 74)
            "corrupt_audio": sample["corrupt_audio"][:],    # Shape: (L_audio, 74)
            "mask_audio": sample["mask_audio"][:],          # Shape: (L_audio,)
            "labels": sample["labels"][:]                   # Shape: (7,)
        }
        return data

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


def test_loader():
    print("Testing BandwidthMoseiDataset loader...")
    dataset = BandwidthMoseiDataset()
    print(f"Total samples loaded: {len(dataset)}")

    # Fetch index 0
    item = dataset[0]
    print(f"\nItem 0 Video ID    : {item['video_id']}")
    print(f"Item 0 Text        : \"{item['text'][:60]}...\"")
    print(f"Text GloVe Shape   : {item['text_glove'].shape}")
    print(f"Clean Audio Shape  : {item['clean_audio'].shape}")
    print(f"Corrupt Audio Shape: {item['corrupt_audio'].shape}")
    print(f"Mask Audio Shape   : {item['mask_audio'].shape}")
    print(f"Labels             : {item['labels']}")
    dataset.close()
    print("\nLoader test completed successfully!")

if __name__ == "__main__":
    test_loader()

