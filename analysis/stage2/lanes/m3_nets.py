"""Lane m3 network architectures (PyTorch, CPU). Every network maps (B, C, T) - or (B, C, F, T) for the 2D CNN -
to one logit per row, g(x). The heads keep coarse TIME POSITION (average-pool to ``n_bins`` bins, then a linear
layer): the hand evidence is the sign of the early yaw, which reverses later in the approach, so a
translation-invariant global-average head would mix evidence of opposite sign (head 'gap' is the ablation).

All networks are small (about 5-25 k parameters), for about 30 training persons' worth of trials."""
from __future__ import annotations

import torch
from torch import nn


class Head(nn.Module):
    """'bins': adaptive average pool to n_bins time bins, flatten, dropout, linear; 'gap': global average."""

    def __init__(self, ch: int, n_bins: int = 8, dropout: float = 0.2, kind: str = 'bins'):
        super().__init__()
        self.kind = kind
        nb = n_bins if kind == 'bins' else 1
        self.pool = nn.AdaptiveAvgPool1d(nb)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(ch * nb, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.pool(h).flatten(1))).squeeze(1)


def conv_bn(cin: int, cout: int, k: int, dilation: int = 1, relu: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv1d(cin, cout, k, padding=dilation * (k - 1) // 2, dilation=dilation, bias=False),
                               nn.BatchNorm1d(cout)]
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class CNN1D(nn.Module):
    def __init__(self, cin: int, width: int = 32, n_bins: int = 8, dropout: float = 0.2, head: str = 'bins'):
        super().__init__()
        self.body = nn.Sequential(conv_bn(cin, width, 7), conv_bn(width, width, 5), conv_bn(width, width, 3))
        self.head = Head(width, n_bins, dropout, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class ResBlock(nn.Module):
    def __init__(self, w: int, k1: int = 5, k2: int = 3):
        super().__init__()
        self.c1 = conv_bn(w, w, k1)
        self.c2 = conv_bn(w, w, k2, relu=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.c2(self.c1(x)))


class ResNet1D(nn.Module):
    def __init__(self, cin: int, width: int = 32, n_bins: int = 8, dropout: float = 0.2, head: str = 'bins'):
        super().__init__()
        self.stem = conv_bn(cin, width, 7)
        self.blocks = nn.Sequential(ResBlock(width), ResBlock(width))
        self.head = Head(width, n_bins, dropout, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class InceptionModule(nn.Module):
    """InceptionTime module, lite: 1x1 bottleneck, three convolutions (9, 5, 3) and a max-pool branch."""

    def __init__(self, cin: int, width: int = 32, bottleneck: int = 8):
        super().__init__()
        b = width // 4
        self.bott = nn.Conv1d(cin, bottleneck, 1, bias=False) if cin > 1 else nn.Identity()
        cb = bottleneck if cin > 1 else cin
        self.convs = nn.ModuleList([nn.Conv1d(cb, b, k, padding=k // 2, bias=False) for k in (9, 5, 3)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(cin, b, 1, bias=False))
        self.bn = nn.BatchNorm1d(4 * b)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bott(x)
        return self.act(self.bn(torch.cat([c(z) for c in self.convs] + [self.pool(x)], dim=1)))


class InceptionLite(nn.Module):
    def __init__(self, cin: int, width: int = 32, n_bins: int = 8, dropout: float = 0.2, head: str = 'bins'):
        super().__init__()
        self.m1 = InceptionModule(cin, width)
        self.m2 = InceptionModule(width, width)
        self.short = nn.Sequential(nn.Conv1d(cin, width, 1, bias=False), nn.BatchNorm1d(width))
        self.act = nn.ReLU()
        self.head = Head(width, n_bins, dropout, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.m2(self.m1(x)) + self.short(x)))


class CausalBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dilation: int):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU()
        self.res = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return self.act(self.bn(h) + self.res(x))


class TCN(nn.Module):
    """Temporal convolutional network: causal dilated convolutions (1, 2, 4, 8), residual connections."""

    def __init__(self, cin: int, width: int = 32, n_bins: int = 8, dropout: float = 0.2, head: str = 'bins'):
        super().__init__()
        chans = [cin, width, width, width, width]
        self.body = nn.Sequential(*[CausalBlock(chans[i], chans[i + 1], 3, 2 ** i) for i in range(4)])
        self.head = Head(width, n_bins, dropout, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class GRUNet(nn.Module):
    """A convolutional front-end, a bidirectional GRU, then the time-binned head over its outputs."""

    def __init__(self, cin: int, width: int = 32, n_bins: int = 8, dropout: float = 0.2, head: str = 'bins',
                 hidden: int = 24):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(cin, width, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(width),
                                   nn.ReLU())                   # halves the sequence: GRU steps are the CPU cost
        self.gru = nn.GRU(width, hidden, batch_first=True, bidirectional=True)
        self.head = Head(2 * hidden, n_bins, dropout, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.front(x).transpose(1, 2))
        return self.head(h.transpose(1, 2))


class TinyTransformer(nn.Module):
    """Patches of 4 samples -> tokens with learned positions -> 2 encoder layers -> time-binned head."""

    def __init__(self, cin: int, width: int = 32, n_bins: int = 8, dropout: float = 0.2, head: str = 'bins',
                 t_len: int = 64, patch: int = 4):
        super().__init__()
        self.patch = nn.Conv1d(cin, width, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, width, t_len // patch))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = Head(width, min(n_bins, t_len // patch), dropout, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch(x) + self.pos
        return self.head(self.enc(h.transpose(1, 2)).transpose(1, 2))


class CNN2D(nn.Module):
    """Time-frequency CNN: input (B, C, F, T) (spectrogram or scalogram planes); the head pools to 2 frequency
    x n_bins time cells, so it keeps where the energy and its sign sit."""

    def __init__(self, cin: int, width: int = 32, n_bins: int = 8, dropout: float = 0.2, head: str = 'bins'):
        super().__init__()
        w1 = width // 2
        self.body = nn.Sequential(
            nn.Conv2d(cin, w1, 3, padding=1, bias=False), nn.BatchNorm2d(w1), nn.ReLU(),
            nn.Conv2d(w1, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((2, n_bins if head == 'bins' else 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 2 * (n_bins if head == 'bins' else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.pool(self.body(x)).flatten(1))).squeeze(1)


ARCHS = {'cnn': CNN1D, 'resnet': ResNet1D, 'inception': InceptionLite, 'tcn': TCN, 'gru': GRUNet,
         'transformer': TinyTransformer, 'cnn2d': CNN2D}
ARCH_LABELS = {'cnn': '1D CNN', 'resnet': 'ResNet1D', 'inception': 'InceptionTime-lite', 'tcn': 'TCN', 'gru': 'GRU',
               'transformer': 'Transformer (tiny)', 'cnn2d': '2D CNN'}


def build_net(arch: str, cin: int, t_len: int, *, width: int = 32, n_bins: int = 8, dropout: float = 0.2,
              head: str = 'bins') -> nn.Module:
    kw = dict(width=width, n_bins=n_bins, dropout=dropout, head=head)
    if arch == 'transformer':
        kw['t_len'] = t_len
    return ARCHS[arch](cin, **kw)


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))
