
from mmsdk import mmdatasdk
import os

dataset_path = "dataset/data"

files = [
    "CMU_MOSEI_Labels.csd",
    "CMU_MOSEI_COVAREP.csd",
    "CMU_MOSEI_OpenFace2.csd",
    "CMU_MOSEI_TimestampedWords.csd",
    "CMU_MOSEI_TimestampedWordVectors.csd",
    "CMU_MOSEI_VisualFacet42.csd",
    "CMU_MOSEI_TimestampedPhones.csd"
]

for file in files:
    path = os.path.join(dataset_path, file)

    print(f"\nChecking: {file}")

    try:
        dataset = mmdatasdk.mmdataset({"test": path})
        print("SUCCESS: File is readable")
    except Exception as e:
        print("ERROR:", e)
