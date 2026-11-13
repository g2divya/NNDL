"""RAVDESS speech emotion recognition: CNN + BiLSTM (speaker-independent split)."""
import glob
import os
import zipfile

import librosa
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, regularizers

# ---------------- Config ----------------
ZIP_PATH = r"C:\Users\Prachi Singh\OneDrive\เอกสาร\Downloads\Audio_Speech_Actors_01-24_16k.zip"
DATA_PATH = r"C:\Users\Prachi Singh\OneDrive\เอกสาร\Downloads\RAVDESS"
CACHE = "ravdess_mel_cache.npz"

SR, DURATION = 16000, 3
N_SAMPLES = SR * DURATION
N_MELS, HOP = 128, 160
NUM_CLASSES = 8
BATCH, EPOCHS, SEED = 32, 80, 42
tf.keras.utils.set_random_seed(SEED)


# ---------------- Features (cached + parallel) ----------------
def extract(path):
    audio, _ = librosa.load(path, sr=SR, duration=DURATION)
    audio = np.pad(audio, (0, max(0, N_SAMPLES - len(audio))))[:N_SAMPLES]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=512, hop_length=HOP, win_length=400,
        n_mels=N_MELS, fmax=8000, power=2)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    parts = os.path.basename(path).split("-")
    return mel_db, int(parts[2]) - 1, int(parts[6].split(".")[0])


def load_dataset():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["X"], d["y"], d["actors"]
    files = glob.glob(os.path.join(DATA_PATH, "**", "*.wav"), recursive=True)
    if not files and os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(DATA_PATH)
        files = glob.glob(os.path.join(DATA_PATH, "**", "*.wav"), recursive=True)
    assert files, "No .wav files found - check paths."
    res = Parallel(n_jobs=-1)(delayed(extract)(f) for f in files)
    X = np.stack([r[0] for r in res])[..., None]
    y = np.array([r[1] for r in res])
    actors = np.array([r[2] for r in res])
    np.savez_compressed(CACHE, X=X, y=y, actors=actors)
    return X, y, actors


X, y, actors = load_dataset()
print("Dataset:", X.shape, y.shape)

# ---------------- Speaker-independent split (16/4/4 actors) ----------------
gss = GroupShuffleSplit(n_splits=1, test_size=4, random_state=SEED)
trv, te = next(gss.split(X, y, groups=actors))
tr, va = next(GroupShuffleSplit(n_splits=1, test_size=4, random_state=SEED)
              .split(X[trv], y[trv], groups=actors[trv]))
tr, va = trv[tr], trv[va]
assert not (set(actors[tr]) & set(actors[te])) and not (set(actors[tr]) & set(actors[va]))

# Standardize with TRAIN statistics only
mean, std = X[tr].mean(), X[tr].std() + 1e-6
X = (X - mean) / std
X_tr, y_tr, X_va, y_va, X_te, y_te = X[tr], y[tr], X[va], y[va], X[te], y[te]

cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_tr)
class_weight = dict(enumerate(cw))


# ---------------- tf.data with SpecAugment (train only) ----------------
def spec_augment(x, label):
    F, T = N_MELS, x.shape[1]
    f = tf.random.uniform([], 0, 20, tf.int32)
    f0 = tf.random.uniform([], 0, F - f, tf.int32)
    t = tf.random.uniform([], 0, 40, tf.int32)
    t0 = tf.random.uniform([], 0, T - t, tf.int32)
    fi, ti = tf.range(F)[:, None, None], tf.range(T)[None, :, None]
    x = tf.where((fi >= f0) & (fi < f0 + f), 0.0, x)
    x = tf.where((ti >= t0) & (ti < t0 + t), 0.0, x)
    return x, label


def make_ds(Xa, ya, train=False):
    ds = tf.data.Dataset.from_tensor_slices((Xa, ya))
    if train:
        ds = ds.shuffle(len(Xa), seed=SEED).map(spec_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds, val_ds, test_ds = make_ds(X_tr, y_tr, True), make_ds(X_va, y_va), make_ds(X_te, y_te)


# ---------------- Model: CNN -> BiLSTM ----------------
def conv_block(x, filters, drop):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.SpatialDropout2D(drop)(x)


inp = layers.Input(shape=X_tr.shape[1:])                  # (128, 301, 1)
x = conv_block(inp, 32, 0.1)
x = conv_block(x, 64, 0.15)
x = conv_block(x, 128, 0.2)                               # (16, 37, 128)
x = layers.Permute((2, 1, 3))(x)                          # (time, freq, ch)
x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                     restore_best_weights=True, mode="max"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
          class_weight=class_weight, callbacks=callbacks, verbose=2)

# ---------------- Accuracy ----------------
_, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
