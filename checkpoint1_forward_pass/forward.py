import numpy as np

# --- Setup ---
np.random.seed(42)

n_samples = 5
n_features = 3
n_outputs = 1

X = np.random.randn(n_samples, n_features)      # shape: (5, 3)
y_true = np.random.randn(n_samples, n_outputs)  # shape: (5, 1)

W = np.random.randn(n_features, n_outputs) * 0.1   # shape: (3, 1)
b = np.zeros((1, n_outputs))                       # shape: (1, 1)


# --- Your job ---

def forward(X, W, b):
    """
    Compute y_pred = X @ W + b
    Return y_pred with shape (n_samples, n_outputs)
    """
    y_pred = X @ W + b
    return y_pred


def mse_loss(y_pred, y_true):
    """
    Compute mean squared error: mean((y_pred - y_true)^2)
    Return a single scalar.
    """
    return np.mean((y_pred - y_true) ** 2)


# --- Run it ---
if __name__ == "__main__":
    y_pred = forward(X, W, b)
    loss = mse_loss(y_pred, y_true)
    print("y_pred shape:", y_pred.shape if y_pred is not None else None)  # should be (5, 1)
    print("loss:", loss)
