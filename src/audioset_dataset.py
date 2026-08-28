import json, ast
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset

with open("audioset_index.json") as f:
    IDX = json.load(f)

CLASSES = ["Scream","Shout","Crying","Explosion","Gunshot","Glass","Siren","Alarm"]
SR = 16000
N_MELS = 64
TIME_FRAMES = 128

class AudioSetDS(Dataset):
    def __init__(self, csv_path, augment=False):
        self.df = pd.read_csv(csv_path)
        self.df["ytid"] = self.df["ytid"].str.strip()
        # keep only rows whose audio exists
        self.df = self.df[self.df["ytid"].isin(IDX)].reset_index(drop=True)
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = IDX[row["ytid"]]
        labels = np.array([row[c] for c in CLASSES], dtype=np.float32)

        y, _ = librosa.load(path, sr=SR, duration=10.0)

        if self.augment:
            if np.random.rand()<0.5: y = y + np.random.randn(len(y))*0.005
            if np.random.rand()<0.4:
                y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9,1.1))
            if np.random.rand()<0.3:
                y = librosa.effects.pitch_shift(y, sr=SR, n_steps=np.random.randint(-2,3))

        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean())/(mel.std()+1e-6)

        if mel.shape[1] < TIME_FRAMES:
            mel = np.pad(mel, ((0,0),(0,TIME_FRAMES-mel.shape[1])))
        else:
            mel = mel[:, :TIME_FRAMES]

        return torch.tensor(mel[np.newaxis], dtype=torch.float32), torch.tensor(labels)
