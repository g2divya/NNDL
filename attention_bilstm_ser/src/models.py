"""Sequence baselines and additive temporal attention model."""
import torch
from torch import nn

class EmotionRNN(nn.Module):
    def __init__(self, input_dim=116, hidden=128, num_classes=8, kind="attention_bilstm", dropout=0.3):
        super().__init__()
        self.kind = kind
        bidirectional = kind != "lstm"
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout)
        out_dim = hidden * (2 if bidirectional else 1)
        self.attention = nn.Sequential(nn.Linear(out_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False)) if kind == "attention_bilstm" else None
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x, lengths=None, return_attention=False):
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
                positions = torch.arange(seq.size(1), device=x.device).unsqueeze(0)
                scores = scores.masked_fill(positions >= lengths.unsqueeze(1), torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        return (logits, weights) if return_attention else logits
