"""LSTM deep learning forecaster for EcoReport AI.

Architecture: 2-layer LSTM → Dense(1)
Training: early stopping, deterministic seeds, GPU auto-detection.
All design rationale documented in docs/design_decisions.md (ADR-010).
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    """Set all random seeds for deterministic LSTM training.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _LSTMNet:
    """Internal PyTorch LSTM network (avoids top-level torch import)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.net = _Net()


class LSTMDataset:
    """Sliding window dataset for LSTM training.

    Converts a time series into overlapping (X, y) windows:
        X[i] = features[i : i + lookback]
        y[i] = target[i + lookback]

    Args:
        features: 2D numpy array (T, n_features).
        targets: 1D numpy array (T,).
        lookback: Number of time steps in each window.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lookback: int,
    ) -> None:
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        for i in range(len(features) - lookback):
            self.X.append(features[i : i + lookback])
            self.y.append(targets[i + lookback])

    def __len__(self) -> int:
        return len(self.X)

    def to_tensors(self):  # type: ignore[return]
        """Convert to PyTorch tensors.

        Returns:
            Tuple of (X_tensor, y_tensor).
        """
        import torch

        X_arr = np.stack(self.X, axis=0).astype(np.float32)
        y_arr = np.array(self.y, dtype=np.float32)
        return torch.from_numpy(X_arr), torch.from_numpy(y_arr)


class LSTMForecaster:
    """LSTM-based time-series forecaster.

    Args:
        lookback: Number of past months as input window.
        hidden_size: Number of LSTM hidden units.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers.
        batch_size: Mini-batch size.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without val loss improvement).
        learning_rate: Adam optimizer learning rate.
        val_split: Fraction of training data held out for early stopping.
        seed: Random seed.
        name: Human-readable model name.
    """

    def __init__(
        self,
        lookback: int = 12,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_size: int = 32,
        max_epochs: int = 200,
        patience: int = 10,
        learning_rate: float = 0.001,
        val_split: float = 0.15,
        seed: int = 42,
        name: str = "LSTM",
    ) -> None:
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.seed = seed
        self.name = name

        self._net: Optional[_LSTMNet] = None
        self._scaler_X = None
        self._scaler_y = None
        self._feature_names: list[str] = []
        self._device: str = "cpu"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LSTMForecaster":
        """Fit the LSTM model.

        Scales features and target, constructs sliding windows, trains with
        early stopping on a held-out validation split.

        Args:
            X_train: Numeric feature DataFrame.
            y_train: Target Series.

        Returns:
            Self.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("PyTorch is required. Run: pip install torch") from exc

        from sklearn.preprocessing import MinMaxScaler

        _set_seeds(self.seed)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("LSTM training on device: %s", self._device)

        # Drop NaN and align
        combined = pd.concat([X_train, y_train], axis=1).dropna()
        y_col = y_train.name or "__target__"
        combined.columns = list(X_train.columns) + [y_col]
        X_clean = combined[list(X_train.columns)].values.astype(np.float32)
        y_clean = combined[y_col].values.astype(np.float32).reshape(-1, 1)
        self._feature_names = list(X_train.columns)

        # Scale
        self._scaler_X = MinMaxScaler()
        self._scaler_y = MinMaxScaler()
        X_scaled = self._scaler_X.fit_transform(X_clean)
        y_scaled = self._scaler_y.fit_transform(y_clean).ravel()

        if len(X_scaled) <= self.lookback + 1:
            logger.warning(
                "LSTM: training set too small (%d rows, lookback=%d). Skipping.",
                len(X_scaled),
                self.lookback,
            )
            return self

        # Build dataset
        dataset = LSTMDataset(X_scaled, y_scaled, self.lookback)
        if len(dataset) < 4:
            logger.warning("LSTM dataset has < 4 samples. Skipping training.")
            return self

        n_val = max(1, int(len(dataset) * self.val_split))
        n_train = len(dataset) - n_val

        X_tensor, y_tensor = dataset.to_tensors()
        train_ds = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
        val_ds = TensorDataset(X_tensor[n_train:], y_tensor[n_train:])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Build net
        net_wrapper = _LSTMNet(
            input_size=X_scaled.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        net = net_wrapper.net.to(self._device)
        self._net = net_wrapper

        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        no_improve_count = 0
        best_state = None

        for epoch in range(self.max_epochs):
            net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

            net.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    val_losses.append(criterion(net(xb), yb).item())
            val_loss = float(np.mean(val_losses))

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                logger.info("LSTM early stopping at epoch %d (val_loss=%.6f).", epoch, val_loss)
                break

        if best_state is not None:
            net.load_state_dict(best_state)

        logger.info(
            "LSTM fitted: %d epochs, best_val_loss=%.6f, features=%d",
            epoch + 1,
            best_val_loss,
            X_scaled.shape[1],
        )
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate LSTM predictions.

        Scales X_test using training scaler, builds sliding windows, runs
        forward pass, and inverse-transforms predictions.

        Args:
            X_test: Feature DataFrame for the test period.

        Returns:
            Array of inverse-scaled predictions, one per test row.
        """
        if self._net is None or self._scaler_X is None:
            logger.warning("LSTM not fitted or training skipped. Returning zeros.")
            return np.zeros(len(X_test))

        import torch

        net = self._net.net
        net.eval()

        X_vals = X_test[self._feature_names].ffill().fillna(0).values.astype(np.float32)
        X_scaled = self._scaler_X.transform(X_vals)

        preds_scaled: list[float] = []

        with torch.no_grad():
            for i in range(len(X_scaled)):
                start = max(0, i - self.lookback + 1)
                window = X_scaled[start : i + 1]
                if len(window) < self.lookback:
                    pad = np.zeros((self.lookback - len(window), window.shape[1]), dtype=np.float32)
                    window = np.concatenate([pad, window], axis=0)
                x_tensor = torch.tensor(window[np.newaxis, :, :], device=self._device)
                pred = net(x_tensor).cpu().numpy().ravel()[0]
                preds_scaled.append(float(pred))

        preds_arr = np.array(preds_scaled).reshape(-1, 1)
        preds_inv = self._scaler_y.inverse_transform(preds_arr).ravel()
        return preds_inv
