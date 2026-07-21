import os
import glob
import librosa
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 1. Target your folder
dataset_path = "./my_drum_kits"
audio_files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)

features = []
file_mapping = []

print(f"Analyzing {len(audio_files)} premium drum samples...")

for file in audio_files:
    try:
        # Load up to 2 seconds to capture long 808 and crash tails
        y, sr = librosa.load(file, duration=2.0)
        
        # Extract MFCCs (Timbre/Texture)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        
        # Extract Spectral Rolloff (Helps separate bright Hi-Hats/Crashes from dark 808s/Kicks)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Combine features into a single complex fingerprint
        fingerprint = np.append(mfcc, [rolloff])
        
        features.append(fingerprint)
        file_mapping.append(file)
    except Exception as e:
        continue

# 2. Upgrade to 7 distinct cluster categories
X = np.array(features)
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10).fit(X)

# 3. Export map
dataset_df = pd.DataFrame({'file_path': file_mapping, 'category_id': kmeans.labels_})
dataset_df.to_csv("advanced_kit_map.csv", index=False)
print("✅ Multi-element drum kit sorted and saved to 'advanced_kit_map.csv'!")
