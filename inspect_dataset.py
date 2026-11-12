from mmsdk import mmdatasdk
import numpy as np
import os

DATA_PATH = "dataset/data"

files = {
    "Labels": "CMU_MOSEI_Labels.csd",
    "Words": "CMU_MOSEI_TimestampedWords.csd",
    "Audio": "CMU_MOSEI_COVAREP.csd",
    "Visual": "CMU_MOSEI_OpenFace2.csd"
}

def inspect_file(name, filename):
    path = os.path.join(DATA_PATH, filename)

    print("\n" + "=" * 65)
    print("DATASET:", name)
    print("=" * 65)

    dataset = mmdatasdk.mmdataset({"sample": path})

    sequence = dataset.computational_sequences["sample"]
    data = sequence.data

    print("Total segments:", len(data))

    # Display first three samples
    for index, (segment_id, item) in enumerate(data.items()):
        if index >= 3:
            break

        print("\nSegment ID:", segment_id)

        features = item["features"]
        intervals = item["intervals"]

        print("Feature shape:", features.shape)
        print("Intervals shape:", intervals.shape)

        print("Feature values:")
        print(features[:5])

        print("Timestamps:")
        print(intervals[:5])

        if name == "Words":
            print("Actual words:")
            for word in features[:10].flatten():
                if isinstance(word, bytes):
                    word = word.decode("utf-8", errors="replace")
                print(word)

# Inspect smaller files first
inspect_file("Labels", files["Labels"])
inspect_file("Words", files["Words"])