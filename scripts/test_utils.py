# scripts/test_utils.py
import numpy as np
from utils import (
    FEAT_SIZE_HOLISTIC,
    to_relative_holistic,
    normalize_dataset,
    save_sequence
)




# 5 sequences, 30 frames, 1662 features
X = np.random.rand(5, 30, FEAT_SIZE_HOLISTIC).astype(np.float32)

# Pick a single sequence
single_seq = X[0]               # (30, 1662)
single_seq_copy = single_seq.copy()


# Test relative conversion

X_rel_single = to_relative_holistic(single_seq_copy)


# Test normalization

Xn, mean, std = normalize_dataset(X)


# Print shapes

print("Shapes check:")
print("  X (batch):", X.shape)
print("  single_seq:", single_seq.shape)
print("  X_rel_single:", X_rel_single.shape)
print("  Xn (normalized batch):", Xn.shape)
print("  mean shape:", mean.shape)
print("  std shape:", std.shape)


# Test saving

save_sequence("data_test", "TEST", 0, single_seq)
print("Saved test sequence to data_test/TEST/TEST_0.npz")
