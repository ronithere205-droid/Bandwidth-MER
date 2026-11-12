# Bandwidth-Aware Multimodal Emotion Recognition (Bandwidth-MER)

This repository contains the simulation pipeline, data extraction routines, and dataset loaders for **Bandwidth-Aware Multimodal Emotion Recognition (MER)** using the **CMU-MOSEI** dataset.

## Research Overview
In distributed multimodal emotion recognition applications (such as tele-health, remote education, and mobile video conferences), modalities experience non-uniform dropouts due to dynamic network fluctuations:
1. **Video (Visual)**: Highest bandwidth footprint; drops first under network constraints.
2. **Audio (Acoustic)**: Moderate footprint (~32–64 kbps); drops intermittently during network degradation.
3. **Text (Transcriptions)**: Negligible footprint (~1 kbps); remains intact as a reliable anchor modality.

This project programmatically simulates realistic, bandwidth-driven temporal missing data on CMU-MOSEI time-series sequences for multimodal compensation and emotion classification.

---

## Python Scripts Overview

| Script | Description |
| :--- | :--- |
| **`bandwidth_simulation_engine.py`** | Implements the 3-state Markov network bandwidth generator (`High`, `Medium`, `Low`) and time-windowed audio corruption engine. |
| **`run_bandwidth_simulation_protocol.py`** | Main 5-step simulation pipeline: reads raw CMU-MOSEI `.csd` sequences, performs temporal alignment, evaluates bandwidth across time windows, applies dropouts, and packages outputs into `.h5`. |
| **`inspect_and_load_h5.py`** | Inspects simulated `.h5` datasets, comparing clean audio, corrupted audio, and binary masks frame-by-frame. |
| **`pytorch_mosei_loader.py`** | PyTorch-ready `Dataset` loader for feeding clean text, corrupted audio, and availability masks into neural networks. |
| **`apply_random_missing_modalities.py`** | Applies uniform random missingness (MCAR) across Text and Audio modalities as an unconstrained baseline. |
| **`create_random_missing_dataset.py`** | Generates static random missing modality CSV benchmarks. |
| **`extract_non_visual_modalities.py`** | Extracts non-visual computational sequences (Text, 74-dim COVAREP, Phonemes, GloVe vectors, Labels) from raw `.csd` archives. |
| **`convert_to_csv.py`** | Utility to extract text words and emotion labels into tabular format. |
| **`inspect_dataset.py`** | Inspects shapes and intervals of raw `.csd` files. |
| **`test_dataset.py`** | Validates readability of CMU-MultimodalSDK `.csd` files. |

---

## Getting Started

### 1. Requirements
```bash
pip install numpy h5py pandas torch
```

### 2. Run the Bandwidth-Aware Simulation
To run the temporal bandwidth dropout simulation across CMU-MOSEI utterances:
```bash
python run_bandwidth_simulation_protocol.py
```

### 3. Inspect the Generated Dataset
```bash
python inspect_and_load_h5.py
```

