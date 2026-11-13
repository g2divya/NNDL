import glob
import os

DATA_PATH = r"C:\Users\Prachi Singh\OneDrive\เอกสาร\Downloads\RAVDESS"

files = glob.glob(
    os.path.join(DATA_PATH, "**", "*.wav"),
    recursive=True
)

print("Total WAV files:", len(files))
print(files[:5])

import os

DATA_PATH = r":\Downloads\RAVDESS"

print("Path exists:", os.path.exists(DATA_PATH))

if os.path.exists(DATA_PATH):
    print("Contents:", os.listdir(DATA_PATH)[:20])
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
audio_files = []
dataset_path = r"C:\Users\Prachi Singh\OneDrive\เอกสาร\Downloads\RAVDESS"
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            audio_files.append(os.path.join(root, file))

print("Total audio files:", len(audio_files))
print(audio_files[:5])    
