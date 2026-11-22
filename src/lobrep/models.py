"""Neural network components: three encoders, the self-supervised (SSL) wrapper, and the task heads.

CONCEPT
-------
An ENCODER reads a window of shape (batch, 50 time steps, 8 features) and compresses it into one
vector of size d ("the representation"/"embedding"). Everything downstream (predicting movement,
predicting volatility, clustering regimes) works on that vector.

SSL (self-supervised learning) trains the encoder WITHOUT labels, using tasks made from the data itself:
  * masked modelling: hide some time steps, ask the model to reconstruct them (like BERT for time series)
  * contrastive learning: two noisy copies of the same window should map close together, and
    different windows far apart (like SimCLR)
Then we FINE-TUNE the pretrained encoder on the labelled task. The research question is whether that
beats training the same network from scratch.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class LSTMEncoder(nn.Module):
    """Recurrent baseline: a 2-layer LSTM; the embedding is the hidden state at the LAST time step."""
    def __init__(self, f, d=64):
        super().__init__()
        self.rnn = nn.LSTM(f, d, num_layers=2, batch_first=True, dropout=0.1)
        self.d = d

    def seq(self, x):          # per-time-step outputs (B, W, d); used by the masked-modelling loss
        return self.rnn(x)[0]

    def forward(self, x):      # one vector per window (B, d)
        return self.seq(x)[:, -1]


class CNNEncoder(nn.Module):
    """Convolutional baseline: three 1-D conv layers with growing dilation (1, 2, 4) so the receptive
    field covers a long stretch of the window (similar in spirit to DeepLOB / WaveNet).
    The embedding is the average over time. padding is chosen to keep the length at W."""
    def __init__(self, f, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(f, d, 5, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=8, dilation=4), nn.GELU())
        self.d = d

    def seq(self, x):
        # Conv1d wants (B, channels, time), our data is (B, time, channels), so transpose around it.
        return self.net(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        return self.seq(x).mean(1)


class TransformerEncoder(nn.Module):
    """Transformer: linear projection to d dims + sinusoidal positional encoding (so attention knows
    the ORDER of time steps) + 2 self-attention layers. Embedding = mean over time.
    `norm_first=True` (pre-LayerNorm) trains more stably on small data."""
    def __init__(self, f, d=64, heads=4, layers=2, max_len=512):
        super().__init__()
        self.inp = nn.Linear(f, d)
        # Fixed sine/cosine position table, as in "Attention Is All You Need".
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)  # saved with the model but not trained
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.d = d

    def seq(self, x):
        h = self.inp(x) + self.pe[: x.size(1)]
        return self.norm(self.enc(h))

    def forward(self, x):
        return self.seq(x).mean(1)


ENCODERS = {"lstm": LSTMEncoder, "cnn": CNNEncoder, "transformer": TransformerEncoder}


def build_encoder(name: str, f: int, d: int = 64) -> nn.Module:
    """Factory: build an encoder by name ('lstm' | 'cnn' | 'transformer')."""
    return ENCODERS[name](f, d)


class SSLModel(nn.Module):
    """Wraps an encoder with the two self-supervised objectives. Total loss =
    lam * masked_reconstruction + (1 - lam) * contrastive.

    mask_ratio: fraction of time-step spans hidden. temp: contrastive 'temperature' (lower = sharper)."""

    def __init__(self, encoder: nn.Module, f: int, mask_ratio=0.3, temp=0.2, lam=0.5):
        super().__init__()
        self.enc, self.mask_ratio, self.temp, self.lam = encoder, mask_ratio, temp, lam
        d = encoder.d
        self.recon = nn.Linear(d, f)  # decoder: per-step embedding -> reconstructed features
        # Projection head for the contrastive loss (thrown away after pretraining, as in SimCLR).
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 32))

    @staticmethod
    def augment(x):
        """Create a 'view' of a window: random per-feature rescale (~10%) plus small Gaussian noise.
        Two different views of the same window form a positive pair."""
        scale = 1 + 0.1 * torch.randn(x.size(0), 1, x.size(2), device=x.device)
        return x * scale + 0.05 * torch.randn_like(x)

    def masked_loss(self, x):
        """Zero out random spans of 3 consecutive steps (~30% of steps), then reconstruct the
        ORIGINAL values; loss (MSE) is measured only on the hidden steps."""
        B, W, _ = x.shape
        m = (torch.rand(B, W // 3 + 1, device=x.device) < self.mask_ratio).repeat_interleave(3, 1)[:, :W]
        xm = x.masked_fill(m.unsqueeze(-1), 0.0)
        pred = self.recon(self.enc.seq(xm))
        return ((pred - x) ** 2)[m].mean() if m.any() else pred.sum() * 0

    def contrastive_loss(self, x):
        """NT-Xent (SimCLR) loss. For a batch of n windows we make 2n views. For each view, its
        partner view is the positive; the other 2n-2 views are negatives. Similarity = cosine / temp."""
        z1 = Fn.normalize(self.proj(self.enc(self.augment(x))), dim=-1)
        z2 = Fn.normalize(self.proj(self.enc(self.augment(x))), dim=-1)
        z = torch.cat([z1, z2])
        sim = z @ z.T / self.temp
        n = x.size(0)
        sim.fill_diagonal_(-1e9)  # a view must not match itself
        tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(x.device)  # index of the partner
        return Fn.cross_entropy(sim, tgt)

    def forward(self, x):
        lm, lc = self.masked_loss(x), self.contrastive_loss(x)
        return self.lam * lm + (1 - self.lam) * lc, lm.item(), lc.item()


class Predictor(nn.Module):
    """Encoder + two small linear heads (multi-task): 3-class price movement and log-volatility."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.enc = encoder
        d = encoder.d
        self.move = nn.Linear(d, 3)
        self.vol = nn.Linear(d, 1)

    def forward(self, x):
        h = self.enc(x)
        return self.move(h), self.vol(h).squeeze(-1)
