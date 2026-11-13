import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =====================================================
# CONFIG
# =====================================================
SEED = 42
DATA_FILE = os.path.expanduser("~/datasets/RAVDESS/ravdess_features.npz")
N_FOLDS = 6              # 24 actors -> 4 actors per test fold
N_VAL_ACTORS = 4         # taken from the training actors of each fold
EPOCHS = 80
BATCH_SIZE = 32          # drop to 16 if you still hit OOM
LR = 3e-4
MODES = ["cnn_only", "cnn_attention"]   # ablation: is attention worth it?
FOLDS_TO_RUN = [0]       # [0] = one-fold smoke test; None = all folds
FEATURES_ARE_TIME_FIRST = None          # None = auto-detect, True/False = force
CROP_FRAMES = "auto"     # "auto" = crop to 98th pct of real clip length, int = fixed, None = no crop
MIXED_PRECISION = True   # set False on old GPUs (pre-Volta)

EMOTIONS = ["Neutral", "Calm", "Happy", "Sad",
            "Angry", "Fearful", "Disgust", "Surprised"]

tf.keras.utils.set_random_seed(SEED)
print("TensorFlow:", tf.__version__)

# GPU: grow memory on demand, optional fp16 compute (halves activation memory)
for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)
if MIXED_PRECISION and tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# =====================================================
# 1. LOAD + VALIDATE + FIX LAYOUT
# =====================================================
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(DATA_FILE)

data = np.load(DATA_FILE)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int32)
actors = data["actors"].astype(np.int32)

assert len(X) == len(y) == len(actors) > 0
assert np.all(np.isfinite(X)), "NaN/inf in features"
assert set(np.unique(y)).issubset(set(range(8)))

if X.ndim == 3:                      # (N, a, b) -> add channel axis
    X = X[..., None]
assert X.ndim == 4, f"Expected 3D/4D features, got {X.shape}"

# Model needs (N, TIME, FREQ, 1). Frames normally outnumber MFCC/mel bins.
time_first = FEATURES_ARE_TIME_FIRST
if time_first is None:
    time_first = X.shape[1] >= X.shape[2]
    print(f"[layout] auto-detected time_first={time_first} from shape {X.shape}. "
          f"Override FEATURES_ARE_TIME_FIRST if this is wrong.")
if not time_first:
    X = np.transpose(X, (0, 2, 1, 3))
print("Features (N, time, freq, ch):", X.shape)

# Label / dataset sanity (RAVDESS: neutral has half the clips of every other class)
counts = np.bincount(y, minlength=8)
print("Class counts:", dict(zip(EMOTIONS, counts.tolist())))
all_actors = sorted(int(a) for a in np.unique(actors))
print("Actors:", len(all_actors), "| clips per actor:",
      sorted(set(int((actors == a).sum()) for a in all_actors)))

# =====================================================
# 2. DETECT PADDING, CROP TO REAL LENGTH
#    Padding = trailing frames identical to the final frame.
# =====================================================
T_full = X.shape[1]
is_pad = np.abs(X[:, :, :, 0] - X[:, -1:, :, 0]).max(axis=2) < 1e-6      # (N, T)
trailing = np.cumprod(is_pad[:, ::-1], axis=1).sum(axis=1)               # >= 1 always
valid_len = (T_full - trailing + 1).astype(np.int32)
print(f"[padding] real length per clip: min={valid_len.min()} "
      f"median={int(np.median(valid_len))} p98={int(np.percentile(valid_len, 98))} "
      f"max={valid_len.max()} | padded fraction={1 - valid_len.mean() / T_full:.2f}")

if CROP_FRAMES == "auto":
    L = int(np.ceil(np.percentile(valid_len, 98) / 8) * 8)
elif CROP_FRAMES is None:
    L = T_full
else:
    L = int(CROP_FRAMES)
L = min(L, T_full)
X = np.ascontiguousarray(X[:, :L])
valid_len = np.minimum(valid_len, L)
time_mask = (np.arange(L)[None, :] < valid_len[:, None]).astype(np.float32)[:, :, None, None]
print("After crop:", X.shape)

# =====================================================
# 3. ACTOR-INDEPENDENT, GENDER-BALANCED FOLDS
#    (RAVDESS: odd actor id = male, even = female)
# =====================================================
rng = np.random.RandomState(SEED)


def make_folds(actor_ids, n_folds):
    males = [a for a in actor_ids if a % 2 == 1]
    females = [a for a in actor_ids if a % 2 == 0]
    rng.shuffle(males)
    rng.shuffle(females)
    folds = [[] for _ in range(n_folds)]
    for i, a in enumerate(males):
        folds[i % n_folds].append(a)
    for i, a in enumerate(females):
        folds[i % n_folds].append(a)
    return folds


def pick_val_actors(train_actors, n_val):
    m = [a for a in train_actors if a % 2 == 1]
    f = [a for a in train_actors if a % 2 == 0]
    rng.shuffle(m)
    rng.shuffle(f)
    half = n_val // 2
    return m[:half] + f[:n_val - half]


def masked_stats(Xa, ma):
    """Per-frequency-bin mean/std over real (non-padded) frames only."""
    n = ma.sum()
    mu = (Xa * ma).sum(axis=(0, 1), keepdims=True) / n
    var = (((Xa - mu) ** 2) * ma).sum(axis=(0, 1), keepdims=True) / n
    return mu, np.sqrt(var) + 1e-6


folds = make_folds(all_actors, N_FOLDS)
print("Test actors per fold:", folds)

# =====================================================
# 4. CUSTOM LAYERS
# =====================================================


@tf.keras.utils.register_keras_serializable(package="ser")
class SpecAugment(layers.Layer):
    """Random time + frequency masking, training only. Input (B, T, F, C), already normalised."""

    def __init__(self, max_time=12, max_freq=6, n_masks=2, **kw):
        super().__init__(**kw)
        self.max_time, self.max_freq, self.n_masks = max_time, max_freq, n_masks

    def _mask(self, x, size, max_w, axis):
        b = tf.shape(x)[0]
        w = tf.random.uniform([b, 1], 0, max_w + 1, dtype=tf.int32)
        room = tf.maximum(size - w, 1)
        start = tf.cast(tf.random.uniform([b, 1]) * tf.cast(room, tf.float32), tf.int32)
        idx = tf.range(size)[None, :]
        m = (idx >= start) & (idx < start + w)
        m = m[:, :, None, None] if axis == 1 else m[:, None, :, None]
        return tf.where(m, tf.zeros_like(x), x)

    def call(self, x, training=None):
        if not training:
            return x
        t, f = tf.shape(x)[1], tf.shape(x)[2]
        for _ in range(self.n_masks):
            x = self._mask(x, t, self.max_time, 1)
            x = self._mask(x, f, self.max_freq, 2)
        return x

    def get_config(self):
        c = super().get_config()
        c.update(max_time=self.max_time, max_freq=self.max_freq, n_masks=self.n_masks)
        return c


@tf.keras.utils.register_keras_serializable(package="ser")
class PositionalEncoding(layers.Layer):
    """Sinusoidal positions. Without this, attention can't tell frame 3 from frame 300."""

    def build(self, input_shape):
        t, d = input_shape[1], input_shape[2]
        pos = np.arange(t)[:, None]
        i = np.arange(d)[None, :]
        ang = pos / np.power(10000.0, (2 * (i // 2)) / d)
        pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
        self.pe = tf.constant(pe[None], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        return x + tf.cast(self.pe, x.dtype)


# =====================================================
# 5. MODEL
# =====================================================


def residual_block(x, filters, dropout=0.1):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Activation("relu")(layers.Add()([x, shortcut]))
    return layers.SpatialDropout2D(dropout)(x)


def build_model(input_shape, use_attention, num_classes=8):
    inp = layers.Input(shape=input_shape)
    x = SpecAugment()(inp)

    x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(x)  # stride 2: ~4x cheaper
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = residual_block(x, 16)
    x = layers.MaxPooling2D(2)(x)
    x = residual_block(x, 32)
    x = layers.MaxPooling2D(2)(x)
    x = residual_block(x, 64)

    # (B, T, F, C) -> (B, T, C): average over frequency
    t, f, c = x.shape[1], x.shape[2], x.shape[3]
    x = layers.AveragePooling2D(pool_size=(1, f))(x)
    x = layers.Reshape((t, c))(x)

    if use_attention:
        x = PositionalEncoding()(x)
        att = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.2)(x, x)
        x = layers.LayerNormalization()(layers.Add()([x, att]))
        ff = layers.Dense(64, activation="relu")(x)
        ff = layers.Dropout(0.2)(ff)
        ff = layers.Dense(c)(ff)
        x = layers.LayerNormalization()(layers.Add()([x, ff]))

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return models.Model(inp, out, name="ser_" + ("attn" if use_attention else "cnn"))


# =====================================================
# 6. CROSS-VALIDATION LOOP
# =====================================================
results = {m: {"acc": [], "f1": [], "y_true": [], "y_pred": []} for m in MODES}

for k, test_actors in enumerate(folds):
    if FOLDS_TO_RUN is not None and k not in FOLDS_TO_RUN:
        continue

    test_mask = np.isin(actors, test_actors)
    train_val_actors = [a for a in all_actors if a not in test_actors]
    val_actors = pick_val_actors(train_val_actors, N_VAL_ACTORS)
    val_mask = np.isin(actors, val_actors)
    train_mask = ~(test_mask | val_mask)

    assert not (set(actors[train_mask]) & set(actors[val_mask]))
    assert not (set(actors[train_mask]) & set(actors[test_mask]))
    assert not (set(actors[val_mask]) & set(actors[test_mask]))

    Xtr, ytr, mtr = X[train_mask], y[train_mask], time_mask[train_mask]
    Xva, yva, mva = X[val_mask], y[val_mask], time_mask[val_mask]
    Xte, yte, mte = X[test_mask], y[test_mask], time_mask[test_mask]

    # Normalise with TRAIN statistics from real frames only; padding is set to 0 (= mean)
    mu, sd = masked_stats(Xtr, mtr)
    Xtr = ((Xtr - mu) / sd) * mtr
    Xva = ((Xva - mu) / sd) * mva
    Xte = ((Xte - mu) / sd) * mte

    classes = np.unique(ytr)
    cw = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw)}

    print(f"\n=== Fold {k + 1}/{N_FOLDS} | test actors {sorted(test_actors)} "
          f"| val {sorted(val_actors)} | train n={len(Xtr)} val n={len(Xva)} test n={len(Xte)}")

    for mode in MODES:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + k)

        model = build_model(Xtr.shape[1:], use_attention=(mode == "cnn_attention"))
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=False,
        )
        hist = model.fit(
            Xtr, ytr,
            validation_data=(Xva, yva),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=12,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=5, min_lr=1e-6, verbose=0),
            ],
            verbose=2,
        )
        pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        results[mode]["acc"].append(acc)
        results[mode]["f1"].append(f1)
        results[mode]["y_true"].append(yte)
        results[mode]["y_pred"].append(pred)
        n_ep = len(hist.history["loss"])
        print(f"  [{mode:14s}] epochs run={n_ep} best val_acc={max(hist.history['val_accuracy']):.3f} "
              f"| TEST acc={acc:.3f} macroF1={f1:.3f} params={model.count_params():,}")
        if n_ep < 15:
            print("  WARNING: very few epochs ran (interrupted or stopped early); "
                  "do not trust this result.")

# =====================================================
# 7. REPORT
# =====================================================
print("\n================ SUMMARY (mean ± std over folds run) ================")
summary = {}
for mode in MODES:
    a, f = np.array(results[mode]["acc"]), np.array(results[mode]["f1"])
    if len(a) == 0:
        continue
    summary[mode] = {"acc_mean": float(a.mean()), "acc_std": float(a.std()),
                     "f1_mean": float(f.mean()), "f1_std": float(f.std()),
                     "fold_acc": a.tolist()}
    print(f"{mode:14s} acc {a.mean():.3f} ± {a.std():.3f} | macroF1 {f.mean():.3f} ± {f.std():.3f}")

if len(MODES) == 2 and len(results[MODES[0]]["acc"]) > 1:
    d = np.array(results[MODES[1]]["acc"]) - np.array(results[MODES[0]]["acc"])
    print(f"\nPer-fold acc diff ({MODES[1]} - {MODES[0]}): {np.round(d, 3)}")
    print(f"Mean diff {d.mean():+.3f} | std {d.std():.3f}  "
          f"(if |mean| < std, the difference is noise)")

with open("cv_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

for mode in MODES:
    if not results[mode]["y_true"]:
        continue
    yt = np.concatenate(results[mode]["y_true"])
    yp = np.concatenate(results[mode]["y_pred"])
    print(f"\nPooled classification report: {mode}")
    print(classification_report(yt, yp, labels=np.arange(8),
                                target_names=EMOTIONS, zero_division=0))
    cm = confusion_matrix(yt, yp, labels=np.arange(8))
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=EMOTIONS).plot(
        cmap="Blues", xticks_rotation=45, ax=ax, values_format="d")
    plt.title(f"{mode} - pooled over {len(results[mode]['acc'])} actor-independent fold(s)")
    plt.tight_layout()
    plt.savefig(f"cm_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close()

print("\nDone. Files: cv_summary.json, cm_<mode>.png")