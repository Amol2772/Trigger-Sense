"""
verify_checkpoint.py — Trigger-Sense

Verifies that a saved model checkpoint reproduces the metrics reported in
the dissertation. Read-only: this script never writes to or overwrites the
checkpoint.

Usage, from the project root:
    python3 src/verify_checkpoint.py

Requires in the working directory:
    best_audioset_sed_v3.pth
    audioset_index.json
    audioset_v2_test.csv
"""

import json
import os
import warnings
import hashlib

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Scream", "Shout", "Crying", "Explosion",
           "Gunshot", "Glass", "Siren", "Alarm"]
SR, N_MELS, TIME_FRAMES, BATCH = 16000, 128, 128, 32
CKPT = "best_audioset_sed_v3.pth"

# Expected values as reported in the dissertation. Update these if the
# reported results are regenerated from a new checkpoint.
CLAIMED = {
    "mAP": 0.634,
    "ci_lo": 0.602,
    "ci_hi": 0.672,
    "micro_tuned": 0.647,
    "macro_tuned": 0.613,
    "thresholds": {
        "Scream": 0.82, "Shout": 0.81, "Crying": 0.92, "Explosion": 0.75,
        "Gunshot": 0.61, "Glass": 0.82, "Siren": 0.80, "Alarm": 0.62,
    },
    "per_class_AP": {
        "Scream": 0.387, "Shout": 0.692, "Crying": 0.781, "Explosion": 0.544,
        "Gunshot": 0.594, "Glass": 0.448, "Siren": 0.859, "Alarm": 0.770,
    },
}


class AudioSetDS(Dataset):
    """Test-time dataset: no augmentation, deterministic."""

    def __init__(self, csv_path, index):
        self.index = index
        self.df = pd.read_csv(csv_path)
        self.df["ytid"] = self.df["ytid"].str.strip()
        self.df = self.df[self.df["ytid"].isin(index)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        labels = np.array([row[c] for c in CLASSES], dtype=np.float32)
        y, _ = librosa.load(self.index[row["ytid"]], sr=SR, duration=10.0)

        # a small number of clips in the corpus load as zero length
        if len(y) < 1024:
            y = np.zeros(SR * 10, dtype=np.float32)

        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS))
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = (np.pad(mel, ((0, 0), (0, TIME_FRAMES - mel.shape[1])))
               if mel.shape[1] < TIME_FRAMES else mel[:, :TIME_FRAMES])

        return (torch.tensor(mel[np.newaxis], dtype=torch.float32),
                torch.tensor(labels))


class CRNN_v3(nn.Module):
    """Final architecture. Block 3 pools frequency only, preserving time."""

    def __init__(self, n, sed=True):
        super().__init__()
        self.sed = sed

        def blk(i, o, pool=(2, 2)):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(pool), nn.Dropout2d(0.1))

        self.cnn = nn.Sequential(
            blk(1, 32, (2, 2)),
            blk(32, 64, (2, 2)),
            blk(64, 128, (2, 1)))
        self.lstm = nn.LSTM(128 * 16, 128, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=0.2)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(256, n)

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x, _ = self.lstm(x)
        x = self.drop(x)
        return self.fc(x) if self.sed else self.fc(x.mean(1))


def file_identity(path):
    """Return size, modification time and MD5 of the checkpoint."""
    size = os.path.getsize(path)
    mtime = pd.Timestamp(os.path.getmtime(path), unit="s")
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return size, mtime, md5


def evaluate(model, index):
    """Run the test set and return probabilities and labels."""
    loader = DataLoader(AudioSetDS("audioset_v2_test.csv", index),
                        batch_size=BATCH, shuffle=False, num_workers=4)
    probs, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pr = torch.sigmoid(model(X.to(DEVICE)).max(1).values)
            probs.append(pr.float().cpu().numpy())
            labels.append(y.numpy())
    return (np.vstack(probs).astype(np.float32),
            np.vstack(labels).astype(np.float32))


def tune_thresholds(probs, labels):
    """Per-class threshold search maximising F1, plus per-class AP."""
    grid = np.arange(0.10, 0.95, 0.01)
    thresholds, aps = {}, []
    for j, cls in enumerate(CLASSES):
        gt, sc = labels[:, j], probs[:, j]
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            f1 = f1_score(gt, (sc > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[cls] = round(float(best_t), 2)
        aps.append(average_precision_score(gt, sc) if gt.sum() > 0 else 0.0)
    return thresholds, aps


def bootstrap_ci(probs, labels, n=5000, seed=0):
    """Bootstrap 95% confidence interval on mAP."""
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n):
        idx = rng.choice(len(labels), len(labels), replace=True)
        boot.append(np.mean([
            average_precision_score(labels[idx, j], probs[idx, j])
            for j in range(len(CLASSES)) if labels[idx, j].sum() > 0]))
    return np.percentile(boot, [2.5, 97.5])


def check(name, got, want, tol=0.005):
    ok = abs(got - want) <= tol
    print(f"  {'MATCH   ' if ok else 'MISMATCH'} {name:14s} "
          f"got {got:.4f}  claimed {want:.4f}  (diff {got - want:+.4f})")
    return ok


def main():
    print("=" * 70)
    print("CHECKPOINT VERIFICATION")
    print("=" * 70)

    if not os.path.exists(CKPT):
        raise SystemExit(f"ERROR: {CKPT} not found in {os.getcwd()}")

    size, mtime, md5 = file_identity(CKPT)
    print(f"\nFile     : {CKPT}")
    print(f"Size     : {size:,} bytes")
    print(f"Modified : {mtime}")
    print(f"MD5      : {md5}")
    print("\n>> If 'Modified' is later than the date the reported results")
    print(">> were generated, the checkpoint has been overwritten.\n")

    with open("audioset_index.json") as f:
        index = json.load(f)

    model = CRNN_v3(8, sed=True).to(DEVICE)
    model.load_state_dict(
        torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Parameters: {n_params:,} (expected 2,914,920)\n")

    print("Running test set...")
    probs, labels = evaluate(model, index)
    print(f"  {probs.shape[0]} clips evaluated\n")

    thresholds, aps = tune_thresholds(probs, labels)
    tuned = np.zeros_like(probs)
    for j, cls in enumerate(CLASSES):
        tuned[:, j] = (probs[:, j] > thresholds[cls]).astype(np.float32)

    mAP = float(np.mean(aps))
    micro = f1_score(labels, tuned, average="micro", zero_division=0)
    macro = f1_score(labels, tuned, average="macro", zero_division=0)
    lo, hi = bootstrap_ci(probs, labels)

    results = []

    print("=" * 70)
    print("HEADLINE METRICS")
    print("=" * 70)
    results.append(check("mAP", mAP, CLAIMED["mAP"]))
    results.append(check("Micro-F1", micro, CLAIMED["micro_tuned"]))
    results.append(check("Macro-F1", macro, CLAIMED["macro_tuned"]))
    print(f"\n  95% CI got [{lo:.4f}, {hi:.4f}]   "
          f"claimed [{CLAIMED['ci_lo']:.4f}, {CLAIMED['ci_hi']:.4f}]")

    print("\n" + "=" * 70)
    print("PER-CLASS AVERAGE PRECISION")
    print("=" * 70)
    for j, cls in enumerate(CLASSES):
        results.append(
            check(cls, aps[j], CLAIMED["per_class_AP"][cls], tol=0.01))

    print("\n" + "=" * 70)
    print("TUNED THRESHOLDS")
    print("=" * 70)
    for cls in CLASSES:
        got, want = thresholds[cls], CLAIMED["thresholds"][cls]
        ok = abs(got - want) <= 0.02
        results.append(ok)
        print(f"  {'MATCH   ' if ok else 'MISMATCH'} {cls:14s} "
              f"got {got:.2f}  claimed {want:.2f}")

    print("\n" + "=" * 70)
    if all(results):
        print("VERDICT: PASS")
        print("The checkpoint reproduces the reported results.")
    else:
        print(f"VERDICT: FAIL — {sum(1 for r in results if not r)} mismatches.")
        print("The checkpoint does not match the reported results. Either")
        print("regenerate the reported numbers from this checkpoint, or")
        print("restore the checkpoint the results were produced with.")
    print("=" * 70)

    out = {
        "file": CKPT, "md5": md5, "size": size, "modified": str(mtime),
        "n_params": n_params,
        "mAP": mAP, "ci": [float(lo), float(hi)],
        "micro_tuned": float(micro), "macro_tuned": float(macro),
        "per_class_AP": {c: float(aps[j]) for j, c in enumerate(CLASSES)},
        "thresholds": thresholds,
        "all_match": bool(all(results)),
    }
    with open("checkpoint_verification.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved checkpoint_verification.json")


if __name__ == "__main__":
    main()
