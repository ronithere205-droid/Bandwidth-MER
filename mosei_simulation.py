import numpy as np

# Non-empty modality combinations: (Text, Audio, Vision)
COMBOS = [
    (1, 0, 0),  # Text only
    (0, 1, 0),  # Audio only
    (0, 0, 1),  # Vision only
    (1, 1, 0),  # Text + Audio
    (1, 0, 1),  # Text + Vision
    (0, 1, 1),  # Audio + Vision
    (1, 1, 1),  # Complete (all 3 present)
]

def random_mcar_mask(rng=None):
    """
    Uniform Missing Completely At Random (MCAR) mask generator.
    Ensures NO PATTERN and NO BIAS towards any modality:
    - Every combination has an equal 1/7 (~14.29%) probability.
    - Each modality has an identical presence rate of 4/7 (~57.14%).
    """
    if rng is None:
        idx = np.random.randint(len(COMBOS))
    else:
        idx = rng.randint(len(COMBOS))
    return COMBOS[idx]

try:
    import torch
    from torch.utils.data import Dataset

    class MoseiRandomMissingDataset(Dataset):
        """
        Multimodal dataset with pattern-free randomly missing modalities.
        """
        def __init__(self, data, mode="train", seed=42):
            self.d = data
            self.mode = mode
            self.rng = np.random.RandomState(seed)
            # Precompute fixed random masks for validation/test for reproducibility
            self.fixed_masks = [random_mcar_mask(self.rng) for _ in range(len(data["label"]))]

        def __len__(self):
            return len(self.d["label"])

        def __getitem__(self, i):
            t = self.d["text"][i]
            a = self.d["audio"][i]
            v = self.d["vision"][i]

            # Uniform random selection with no priority/pattern
            if self.mode == "train":
                m = random_mcar_mask()
            else:
                m = self.fixed_masks[i]

            # Apply mask (0 zeroes out the missing modality)
            t = t * m[0]
            a = a * m[1]
            v = v * m[2]

            mask_tensor = torch.tensor(m, dtype=torch.float)
            label = self.d["label"][i]

            return t, a, v, mask_tensor, label

except ImportError:
    pass
