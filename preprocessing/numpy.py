
# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)
actors = np.array(actors)

# Check dataset before splitting
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Actors shape:", actors.shape)
print("Audio files:", len(files))

assert len(X) > 0, "X is empty. Check your audio loading and feature extraction."
assert len(X) == len(y) == len(actors), "Dataset lengths do not match."

# First split: training + validation vs testing
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=4,
    random_state=42
)

train_val_idx, test_idx = next(
    split1.split(X, y, groups=actors)
)

# Second split: training vs validation
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=4,
    random_state=42
)

train_idx, val_idx = next(
    split2.split(
        X[train_val_idx],
        y[train_val_idx],
        groups=actors[train_val_idx]
    )
)

# Map second split indices back to original dataset
train_idx = train_val_idx[train_idx]
val_idx = train_val_idx[val_idx]

# Create datasets
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("Training actors:", np.unique(actors[train_idx]))
print("Validation actors:", np.unique(actors[val_idx]))
print("Testing actors:", np.unique(actors[test_idx]))