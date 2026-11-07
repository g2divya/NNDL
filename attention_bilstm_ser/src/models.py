"""Sequence baselines and additive temporal attention model."""
from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path
import numpy as np
from scipy.special import expit as sigmoid, softmax

VALID_KINDS = ("lstm", "bilstm", "attention_bilstm")

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False


if HAS_TORCH:
    class EmotionRNN(nn.Module):
        def __init__(self, input_dim=116, hidden=128, num_classes=8, kind="attention_bilstm", dropout=0.3):
            super().__init__()
            if kind not in VALID_KINDS:
                raise ValueError(f"Unsupported model kind {kind!r}; expected one of {VALID_KINDS}")
            self.kind = kind
            self.input_dim = int(input_dim)
            self.hidden = int(hidden)
            self.num_classes = int(num_classes)
            bidirectional = kind != "lstm"
            self.rnn = nn.LSTM(
                input_dim,
                hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout,
            )
            out_dim = hidden * (2 if bidirectional else 1)
            self.attention = (
                nn.Sequential(nn.Linear(out_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
                if kind == "attention_bilstm"
                else None
            )
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

        def forward(self, x, lengths=None, return_attention=False):
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim != 3 or x.size(-1) != self.input_dim:
                raise ValueError(f"Expected input tensor of shape [batch, time, {self.input_dim}], got {tuple(x.shape)}")
            seq, (hidden_state, _) = self.rnn(x)
            if self.attention is None:
                if self.rnn.bidirectional:
                    # Concatenate the final forward and backward states from the top layer.
                    pooled = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
                else:
                    pooled = hidden_state[-1]
                weights = None
            else:
                scores = self.attention(seq).squeeze(-1)
                if lengths is not None:
                    lengths = lengths.to(device=x.device)
                    positions = torch.arange(seq.size(1), device=x.device).unsqueeze(0)
                    scores = scores.masked_fill(
                        positions >= lengths.unsqueeze(1), torch.finfo(scores.dtype).min
                    )
                weights = torch.softmax(scores, dim=1)
                pooled = torch.sum(seq * weights.unsqueeze(-1), dim=1)
            logits = self.classifier(pooled)
            return (logits, weights) if return_attention else logits
else:
    EmotionRNN = None  # type: ignore


class NumpyEmotionRNN:
    """Portable NumPy/SciPy inference engine for trained EmotionRNN checkpoints."""

    def __init__(self, checkpoint: dict):
        kind = checkpoint.get("kind", "attention_bilstm")
        if kind not in VALID_KINDS:
            raise ValueError(f"Unsupported model kind {kind!r}; expected one of {VALID_KINDS}")
        self.checkpoint = checkpoint
        self.state_dict = checkpoint["state_dict"]
        self.kind = kind
        self.hidden = int(checkpoint.get("hidden", 128))
        self.input_dim = int(checkpoint.get("input_dim", 116))
        self.labels = list(checkpoint.get("labels", []))
        self.num_classes = len(self.labels) if self.labels else int(self.state_dict["classifier.1.bias"].shape[0])

    def eval(self):
        return self

    @staticmethod
    def _lstm_direction(
        x_seq: np.ndarray,
        w_ih: np.ndarray,
        w_hh: np.ndarray,
        b_ih: np.ndarray,
        b_hh: np.ndarray,
        reverse: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_steps, _ = x_seq.shape
        hidden = w_hh.shape[1]
        h = np.zeros(hidden, dtype=np.float32)
        c = np.zeros(hidden, dtype=np.float32)
        out = np.zeros((t_steps, hidden), dtype=np.float32)
        indices = range(t_steps - 1, -1, -1) if reverse else range(t_steps)
        x_proj = x_seq @ w_ih.T + (b_ih + b_hh)
        for t in indices:
            gates = x_proj[t] + h @ w_hh.T
            i_gate = sigmoid(gates[0:hidden])
            f_gate = sigmoid(gates[hidden : 2 * hidden])
            g_gate = np.tanh(gates[2 * hidden : 3 * hidden])
            o_gate = sigmoid(gates[3 * hidden : 4 * hidden])
            c = f_gate * c + i_gate * g_gate
            h = o_gate * np.tanh(c)
            out[t] = h
        return out, h

    def _forward_single(
        self, x: np.ndarray, length: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected 2D feature array of shape [time, {self.input_dim}], got {x.shape}")
        sd = self.state_dict
        bidirectional = self.kind != "lstm"
        seq = x.astype(np.float32)
        last_f, last_b = None, None
        for layer in range(2):
            out_f, last_f = self._lstm_direction(
                seq,
                sd[f"rnn.weight_ih_l{layer}"],
                sd[f"rnn.weight_hh_l{layer}"],
                sd[f"rnn.bias_ih_l{layer}"],
                sd[f"rnn.bias_hh_l{layer}"],
                reverse=False,
            )
            if bidirectional:
                out_b, last_b = self._lstm_direction(
                    seq,
                    sd[f"rnn.weight_ih_l{layer}_reverse"],
                    sd[f"rnn.weight_hh_l{layer}_reverse"],
                    sd[f"rnn.bias_ih_l{layer}_reverse"],
                    sd[f"rnn.bias_hh_l{layer}_reverse"],
                    reverse=True,
                )
                seq = np.concatenate([out_f, out_b], axis=1)
            else:
                seq = out_f

        if self.kind == "attention_bilstm" and "attention.0.weight" in sd:
            att_h = np.tanh(seq @ sd["attention.0.weight"].T + sd["attention.0.bias"])
            scores = (att_h @ sd["attention.2.weight"].T).squeeze(-1)
            if length is not None:
                mask = np.arange(seq.shape[0]) >= int(length)
                scores = np.where(mask, np.finfo(np.float32).min, scores)
            weights = softmax(scores).astype(np.float32)
            pooled = np.sum(seq * weights[:, None], axis=0)
        else:
            weights = None
            pooled = np.concatenate([last_f, last_b], axis=0) if bidirectional else last_f

        logits = (pooled @ sd["classifier.1.weight"].T + sd["classifier.1.bias"]).astype(np.float32)
        probs = softmax(logits).astype(np.float32)
        return logits, probs, weights

    def predict(
        self, feat: np.ndarray, return_attention: bool = True, lengths: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Run forward pass on [T, D] or [B, T, D] features; returns (logits, probs, attention_weights)."""
        x = np.asarray(feat, dtype=np.float32)
        if x.ndim == 2:
            length = int(lengths[0]) if lengths is not None else None
            logits, probs, weights = self._forward_single(x, length=length)
            return (logits, probs, weights) if return_attention else (logits, probs, None)
        if x.ndim == 3:
            if x.shape[0] == 1:
                length = int(lengths[0]) if lengths is not None else None
                logits, probs, weights = self._forward_single(x[0], length=length)
                return (logits, probs, weights) if return_attention else (logits, probs, None)
            logits_list, probs_list, weights_list = [], [], []
            for i in range(x.shape[0]):
                l_i = int(lengths[i]) if lengths is not None else None
                lg, pr, wt = self._forward_single(x[i], length=l_i)
                logits_list.append(lg)
                probs_list.append(pr)
                if wt is not None:
                    weights_list.append(wt)
            logits_b = np.stack(logits_list, axis=0)
            probs_b = np.stack(probs_list, axis=0)
            weights_b = np.stack(weights_list, axis=0) if weights_list else None
            return (logits_b, probs_b, weights_b) if return_attention else (logits_b, probs_b, None)
        raise ValueError(f"Expected 2D or 3D input array, got shape {x.shape}")


def load_checkpoint_numpy(pt_path: str | Path) -> dict:
    """Load a PyTorch zip checkpoint directly into NumPy arrays without requiring torch."""
    with zipfile.ZipFile(pt_path, "r") as zf:
        prefix = zf.namelist()[0].split("/")[0]
        storages: dict[str, object] = {}

        class _StorageHolder:
            def __init__(self, key: str, dtype: np.dtype):
                raw = zf.read(f"{prefix}/data/{key}")
                self.arr = np.frombuffer(raw, dtype=dtype).copy()

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    def _rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        count = int(np.prod(size))
                        arr = storage.arr[storage_offset : storage_offset + count]
                        return arr.reshape(size)
                    return _rebuild
                if module == "torch" and "Storage" in name:
                    dtype_map = {
                        "FloatStorage": np.float32,
                        "LongStorage": np.int64,
                        "DoubleStorage": np.float64,
                        "HalfStorage": np.float16,
                    }
                    return dtype_map.get(name, np.float32)
                return super().find_class(module, name)

            def persistent_load(self, pid):
                _, dtype, key, _, _ = pid
                if key not in storages:
                    storages[key] = _StorageHolder(key, dtype)
                return storages[key]

        return _Unpickler(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()


def load_inference_model(pt_path: str | Path) -> NumpyEmotionRNN | None:
    """Load a portable inference model from a .pt checkpoint (works with or without PyTorch)."""
    path = Path(pt_path)
    if not path.is_file():
        return None
    if HAS_TORCH:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt_np = {
            k: ({sk: sv.detach().cpu().numpy() for sk, sv in v.items()} if k == "state_dict" else v)
            for k, v in ckpt.items()
        }
        return NumpyEmotionRNN(ckpt_np)
    ckpt_np = load_checkpoint_numpy(path)
    return NumpyEmotionRNN(ckpt_np)
