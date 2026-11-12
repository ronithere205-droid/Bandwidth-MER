
from mmsdk import mmdatasdk
import pandas as pd
import os

DATA_PATH = "dataset/data"

# Load labels
labels = mmdatasdk.mmdataset({
    "labels": os.path.join(DATA_PATH, "CMU_MOSEI_Labels.csd")
})

# Load words
words = mmdatasdk.mmdataset({
    "words": os.path.join(DATA_PATH, "CMU_MOSEI_TimestampedWords.csd")
})

label_data = labels.computational_sequences["labels"].data
word_data = words.computational_sequences["words"].data

emotion_names = [
    "Sentiment", "Happy", "Sad", "Angry",
    "Fear", "Disgust", "Surprise"
]

rows = []

# Match each label interval with its corresponding words
for video_id, label_item in label_data.items():

    if video_id not in word_data:
        continue

    label_features = label_item["features"]
    label_intervals = label_item["intervals"]

    word_features = word_data[video_id]["features"]
    word_intervals = word_data[video_id]["intervals"]

    for i, (start, end) in enumerate(label_intervals):

        sentence_words = []

        for j, (word_start, word_end) in enumerate(word_intervals):

            # Select words overlapping this label interval
            if word_start < end and word_end > start:

                word = word_features[j][0]

                if isinstance(word, bytes):
                    word = word.decode("utf-8", errors="replace")

                if word.strip() and word.lower() != "sp":
                    sentence_words.append(word)

        sentence = " ".join(sentence_words)

        values = label_features[i]

        row = {
            "Video_ID": video_id,
            "Start_Time": start,
            "End_Time": end,
            "Text": sentence
        }

        for k, emotion in enumerate(emotion_names):
            row[emotion] = float(values[k])

        rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Save CSV
output_file = "CMU_MOSEI_Text_Labels.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\nCSV CREATED SUCCESSFULLY!")
print("Total utterances:", len(df))
print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

print("\nSaved as:", output_file)
