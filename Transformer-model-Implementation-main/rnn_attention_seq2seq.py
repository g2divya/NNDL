import argparse
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
VOCAB = [str(i) for i in range(10)] + ["<pad>", "<sos>", "<eos>"]
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}


def make_sequence(length):
    return [random.randint(0, 9) for _ in range(length)]


class ReverseDataset(Dataset):
    def __init__(self, samples=4000, min_len=1, max_len=6):
        self.samples = []
        for _ in range(samples):
            seq = make_sequence(random.randint(min_len, max_len))
            self.samples.append((seq, list(reversed(seq))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, trg = self.samples[idx]
        return src, trg


def collate_batch(batch):
    src_seqs, trg_seqs = zip(*batch)
    max_src_len = max(len(seq) for seq in src_seqs)
    max_decoder_len = max(len(seq) + 1 for seq in trg_seqs)
    max_len = max(max_src_len, max_decoder_len)

    src_batch = []
    trg_batch = []

    for src, trg in zip(src_seqs, trg_seqs):
        padded_src = src + [PAD_TOKEN] * (max_len - len(src))
        tgt_input = [SOS_TOKEN] + trg
        target = trg + [EOS_TOKEN]
        padded_tgt_input = tgt_input + [PAD_TOKEN] * (max_len - len(tgt_input))
        padded_target = target + [PAD_TOKEN] * (max_len - len(target))
        src_batch.append(padded_src)
        trg_batch.append((padded_tgt_input, padded_target))

    src_tensor = torch.tensor(src_batch, dtype=torch.long)
    tgt_input = torch.tensor([item[0] for item in trg_batch], dtype=torch.long)
    trg_target = torch.tensor([item[1] for item in trg_batch], dtype=torch.long)

    return src_tensor, tgt_input, trg_target


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.out = nn.Linear(hidden_dim * 2, output_size)
        self.scale = math.sqrt(hidden_dim)

    def forward_step(self, decoder_input, hidden, encoder_outputs):
        embedded = self.embedding(decoder_input)
        decoder_outputs, hidden = self.gru(embedded, hidden)

        query = decoder_outputs
        scores = torch.bmm(query, encoder_outputs.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, encoder_outputs)
        combined = torch.cat([query, context], dim=-1)
        logits = self.out(combined)
        return logits, hidden, attention_weights

    def forward(self, decoder_input, hidden, encoder_outputs):
        outputs = []
        weights = []

        for step in range(decoder_input.size(1)):
            current_input = decoder_input[:, step:step + 1]
            logits, hidden, attn = self.forward_step(current_input, hidden, encoder_outputs)
            outputs.append(logits)
            weights.append(attn)

        logits = torch.cat(outputs, dim=1)
        attn_weights = torch.cat(weights, dim=1)
        return logits, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = EncoderRNN(vocab_size, embedding_dim, hidden_dim)
        self.decoder = DecoderRNN(vocab_size, embedding_dim, hidden_dim)

    def forward(self, src, trg_input):
        encoder_outputs, hidden = self.encoder(src)
        decoder_logits, attention = self.decoder(trg_input, hidden, encoder_outputs)
        return decoder_logits, attention

    def predict(self, src, max_len=8):
        self.eval()
        with torch.no_grad():
            src = src.unsqueeze(0)
            encoder_outputs, hidden = self.encoder(src)
            decoder_input = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=src.device)
            generated = []

            for _ in range(max_len):
                logits, hidden, _ = self.decoder.forward_step(decoder_input, hidden, encoder_outputs)
                next_token = logits[0, -1].argmax(dim=-1).item()
                generated.append(next_token)
                if next_token == EOS_TOKEN:
                    break
                decoder_input = torch.tensor([[next_token]], dtype=torch.long, device=src.device)

            return generated


def training_loop(model, loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for src_batch, tgt_input_batch, tgt_target_batch in loader:
            src_batch = src_batch.to(device)
            tgt_input_batch = tgt_input_batch.to(device)
            tgt_target_batch = tgt_target_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(src_batch, tgt_input_batch)
            logits = logits.reshape(-1, logits.size(-1))
            targets = tgt_target_batch.reshape(-1)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="RNN-based attention sequence-to-sequence model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ReverseDataset(samples=args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    model = Seq2Seq(vocab_size=len(VOCAB), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    training_loop(model, loader, optimizer, criterion, device, args.epochs)

    examples = [
        [3, 1, 5, 8],
        [9, 0, 2],
        [4, 7, 1, 6, 2],
    ]

    print("\nSample predictions:")
    for seq in examples:
        input_tokens = [int(token) for token in seq]
        prediction = model.predict(torch.tensor(input_tokens, dtype=torch.long, device=device), max_len=len(seq) + 2)
        prediction = [ID_TO_TOKEN.get(token, "?") for token in prediction if token != EOS_TOKEN]
        print(f"Input: {seq} -> Prediction: {prediction}")


if __name__ == "__main__":
    main()
