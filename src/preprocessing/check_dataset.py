import numpy as np

train = np.load("data/processed/train.npz")
test = np.load("data/processed/test.npz")

print("Train:", train["X_len"].shape, train["X_iat"].shape, train["y"].shape)
print("Test :", test["X_len"].shape, test["X_iat"].shape, test["y"].shape)
print("Train classes:", sorted(set(train["y"].tolist())))