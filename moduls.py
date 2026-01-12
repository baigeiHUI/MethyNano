import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
class ConvBNGeLU1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 bn_eps=1e-5, bn_momentum=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.act = nn.GELU()
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):  # x: [B, C, T]
        return self.act(self.bn(self.conv(x)))


class ConvBNGeLU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3),
                 stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False,
                 bn_eps=1e-5, bn_momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.act = nn.GELU()
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):  # x: [B, C, H, W]
        return self.act(self.bn(self.conv(x)))
class SEBlock2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class ResBlock2d(nn.Module):
    def __init__(self, ch: int, k: tuple[int,int]=(3,3), drop: float=0.0):
        super().__init__()
        self.conv1 = ConvBNGeLU2d(ch, ch, kernel_size=k, padding=(k[0]//2, k[1]//2))
        self.conv2 = ConvBNGeLU2d(ch, ch, kernel_size=k, padding=(k[0]//2, k[1]//2))
        self.se = SEBlock2d(ch)
        self.drop = nn.Dropout2d(drop)
    def forward(self, x):
        h = self.conv1(x)
        h = self.drop(self.conv2(h))
        h = self.se(h)
        return x + h
def sinusoid_position_encoding(length: int, dim: int, device=None, dtype=None):
    device = device or torch.device("cpu")
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    position = torch.arange(0, length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class GatedResidual(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            GEGLU(dim, dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.ff(x)
class SignalEncoder(nn.Module):

    def __init__(self, use_dw_res: bool = True, base: int = 256,
                 out_dim: int = 256, drop: float = 0.1, use_stats: bool = True):
        super().__init__()
        self.use_dw_res = use_dw_res
        self.use_stats = use_stats


        self.dw_1x5 = ConvBNGeLU1d(13, 13, kernel_size=5, padding=2, groups=13)


        self.br1 = ConvBNGeLU2d(1, base, kernel_size=(1, 3), padding=(0, 1))
        self.br2 = ConvBNGeLU2d(1, base, kernel_size=(1, 5), padding=(0, 2))
        self.br3 = ConvBNGeLU2d(1, base, kernel_size=(1, 7), padding=(0, 3))
        self.down_t = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.merge_conv = ConvBNGeLU2d(base * 3, base * 3, kernel_size=(1, 3), padding=(0, 1))


        feat_dim = base * 3
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
        )

        if self.use_stats:
            self.stats_film = nn.Linear(3, feat_dim * 2)  # 每个位点 3 维 stats → 2 * feat_dim

    def forward(self, sig, stats):
        """
        sig:   [B, 13, T]
        stats: [B, 13, 3] or [B, 39]
        """
        B, C, T = sig.shape
        assert C == 13, f"Expect 13 channels (k-mer positions), got {C}"

        # 1) per-base 1D conv
        x = self.dw_1x5(sig)  # [B,13,T]
        if self.use_dw_res:
            x = x + sig


        #  [B,1,H=13,W=T]
        x2d = x.unsqueeze(1)  # [B,1,13,T]
        a = self.down_t(self.br1(x2d))  # [B, base, 13, T']
        b = self.down_t(self.br2(x2d))  # [B, base, 13, T']
        c = self.down_t(self.br3(x2d))  # [B, base, 13, T']

        x2d = torch.cat([a, b, c], dim=1)         # [B, base*3, 13, T']
        x2d = self.merge_conv(x2d)                # [B, base*3, 13, T']


        mean_t = x2d.mean(dim=-1)                 # [B, base*3, 13]
        max_t  = x2d.max(dim=-1).values           # [B, base*3, 13]
        feat = 0.5 * (mean_t + max_t)             # [B, base*3, 13]

        # reshape
        feat = feat.permute(0, 2, 1)              # [B,13, base*3]

        # 4) FiLM
        if self.use_stats and stats is not None:
            if stats.dim() == 2:                  # [B,39] -> [B,13,3]
                stats = stats.view(B, 13, 3)
            elif stats.dim() == 3:
                assert stats.size(1) == 13 and stats.size(2) == 3, \
                    f"expect stats [B,13,3], got {stats.shape}"

            film = self.stats_film(stats)         # [B,13, 2*feat_dim]
            gamma, beta = film.chunk(2, dim=-1)   # [B,13,feat_dim]
            feat = gamma * feat + beta            # FiLM


        tokens = self.proj(feat)                  # [B,13,out_dim]
        L = tokens.size(1)
        pos = sinusoid_position_encoding(L, tokens.size(-1),
                                         device=tokens.device,
                                         dtype=tokens.dtype)  # [L,D]
        tokens = tokens + pos.unsqueeze(0)        # [B,13,out_dim]
        h_sig = tokens.mean(dim=1)                # [B,out_dim]
        return tokens, h_sig




class SequenceEncoder(nn.Module):

    def __init__(self, dropout: float = 0.1, proj_dim: int = 256,  pad_id: int = 0,use_stats_in_seq: bool = True):
        super().__init__()
        self.pad_id = pad_id
        self.use_stats_in_seq = use_stats_in_seq
        if self.use_stats_in_seq:
            self.stats2emb = nn.Linear(3, 128)
        self.embed = nn.Embedding(5, 128, padding_idx=pad_id,)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=192,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, stats: Optional[torch.Tensor] = None):

        emb = self.embed(x)  # [B, L, 128]

        if self.use_stats_in_seq and stats is not None:
            B, L = x.shape
            if stats.dim() == 2:  # [B,39] -> [B,L,3]
                stats = stats.view(B, L, 3)
            elif stats.dim() == 3:
                assert stats.size(1) == L and stats.size(2) == 3, \
                    f"expect stats [B,{L},3], got {stats.shape}"
            stats_emb = self.stats2emb(stats)  # [B,L,128]
            emb = emb + stats_emb

        out, _ = self.lstm(emb)  # [B, L, 384]
        tokens = self.proj(out)  # [B, L, proj_dim]
        mask = (x != self.pad_id).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1)
        h_seq = (tokens * mask).sum(dim=1) / denom
        return tokens, h_seq

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x / keep * mask

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    def forward(self, x):
        a, g = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(g)
class CrossAttnBlock(nn.Module):

    def __init__(self, dimension=512, n_heads=8, ffn_ratio=4, dropout=0.1, droppath: float = 0.0):
        super().__init__()
        self.ln_q_seq = nn.LayerNorm(dimension)
        self.ln_kv_seq = nn.LayerNorm(dimension)
        self.ln_q_sig = nn.LayerNorm(dimension)
        self.ln_kv_sig = nn.LayerNorm(dimension)
        self.attn_seq = nn.MultiheadAttention(dimension, n_heads, dropout=dropout, batch_first=True)
        self.attn_sig = nn.MultiheadAttention(dimension, n_heads, dropout=dropout, batch_first=True)
        hidden = dimension * ffn_ratio
        self.ffn_seq = nn.Sequential(
            nn.LayerNorm(dimension),
            GEGLU(dimension, hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )
        self.ffn_sig = nn.Sequential(
            nn.LayerNorm(dimension),
            GEGLU(dimension, hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )
        self.dp1 = DropPath(droppath)
        self.dp2 = DropPath(droppath)
        self.dp3 = DropPath(droppath)
        self.dp4 = DropPath(droppath)
    def forward(self, tokens_seq, tokens_sig, key_padding_mask_seq: Optional[torch.Tensor]=None, key_padding_mask_sig: Optional[torch.Tensor]=None):
        # seq <- sig
        q = self.ln_q_seq(tokens_seq); k = self.ln_kv_seq(tokens_sig)
        attn_seq, _ = self.attn_seq(q, k, k, key_padding_mask=key_padding_mask_sig)
        z_seq = tokens_seq + self.dp1(attn_seq)
        z_seq = z_seq + self.dp2(self.ffn_seq(z_seq))
        # sig <- seq
        q2 = self.ln_q_sig(tokens_sig); k2 = self.ln_kv_sig(tokens_seq)
        attn_sig, _ = self.attn_sig(q2, k2, k2, key_padding_mask=key_padding_mask_seq)
        z_sig = tokens_sig + self.dp3(attn_sig)
        z_sig = z_sig + self.dp4(self.ffn_sig(z_sig))
        return z_seq, z_sig


class StatsEncoder(nn.Module):

    def __init__(self, in_dim=39, hidden=128, out_dim=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, stats):
        if stats.dim() == 3:           # [B,5,3] -> [B,15]
            B = stats.size(0)
            stats = stats.reshape(B, -1)
        return self.proj(stats)        # [B,256]


class GlobalPool(nn.Module):
    def __init__(self, mode="mean"):
        super().__init__()

    def forward(self, x, mask = None):
        mean = x.mean(dim=1)  # [B, D]
        maxv = x.max(dim=1).values  # [B, D]
        return 0.5 * (mean + maxv)
class CenterPool(nn.Module):

    def __init__(self, alpha: float = 0.6):
        super().__init__()
        self.alpha = float(alpha)
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        device, dtype = x.device, x.dtype
        idx = torch.arange(L, device=device, dtype=dtype)
        center = (L - 1) / 2.0
        w = torch.exp(-((idx - center) ** 2) / (2 * (0.4 * L) ** 2))
        w = (w / w.sum()).view(1, L, 1)
        wmean = (x * w).sum(dim=1)          # [B, D]
        maxv  = x.max(dim=1).values         # [B, D]
        return self.alpha * wmean + (1 - self.alpha) * maxv

class CenterSqueeze2d(nn.Module):

    def __init__(self, H: int = 13, sigma_ratio: float = 0.3):
        super().__init__()
        idx = torch.arange(H, dtype=torch.float32)
        center = (H - 1) / 2.0
        w = torch.exp(-((idx - center) ** 2) / (2 * (sigma_ratio * H) ** 2))
        w = w / w.sum()
        self.register_buffer("w", w.view(1, 1, H, 1))  # [1,1,H,1]
    def forward(self, x):
        # x: [B, C, H, T]
        # return [B, C, 1, T]
        return (x * self.w).sum(dim=2, keepdim=True)
class CrossModalFuser(nn.Module):

    def __init__(self, dimension=256, n_heads=8, dropout=0.1, use_stats=True, droppath: float=0.0):
        super().__init__()
        self.use_stats = use_stats
        self.coattn = CrossAttnBlock(dimension=dimension, n_heads=n_heads, dropout=dropout, droppath=droppath)

        self.gr_seq = GatedResidual(dimension, dropout=dropout)
        self.gr_sig = GatedResidual(dimension, dropout=dropout)
        self.pool = GlobalPool()
        self.pool_center = CenterPool(alpha=0.6)
        fuse_in = dimension * (3 if use_stats else 2)
        if use_stats:
            self.stats_enc = StatsEncoder(in_dim=39, hidden=256, out_dim=dimension, dropout=dropout)


        self.fuse = nn.Sequential(
            nn.LayerNorm(fuse_in),
            GEGLU(fuse_in, 512),
            nn.Dropout(dropout),
            nn.Linear(512, dimension),
        )

    def forward(self, tokens_seq, tokens_sig, stats=None):
        z_seq, z_sig = self.coattn(tokens_seq, tokens_sig)
        z_seq = self.gr_seq(z_seq)
        z_sig = self.gr_sig(z_sig)
        g_seq = self.pool_center(z_seq)
        g_sig = self.pool_center(z_sig)
        if self.use_stats:
            assert stats is not None, "When use_stats=True, 'stats' must be provided"
            g_stat = self.stats_enc(stats)
            h = torch.cat([g_seq, g_sig, g_stat], dim=-1)
        else:
            h = torch.cat([g_seq, g_sig], dim=-1)
        h_fused = self.fuse(h)
        return h_fused


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=256, hid_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
        )
    def forward(self, h):
        x = self.net(h)
        z = F.normalize(x.float(), dim=-1, eps=1e-6)
        return z

class ClassificationHead(nn.Module):
    def __init__(self, in_dim=256, hidden=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden , 2),
        )
    def forward(self, h):
        logits = self.mlp(h)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

import torch
import torch.nn as nn
import torch.nn.functional as F

class MethyNano(nn.Module):
    def __init__(
        self,
        use_stats: bool = True,
        with_projection: bool = False,
        with_classification: bool = True,
        dimension: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        base_sig: int = 256,
    ):
        super().__init__()
        self.signal_encoder = SignalEncoder(out_dim=dimension, drop=dropout,
                                            use_stats=use_stats, base=base_sig)
        self.sequence_encoder = SequenceEncoder(dropout=dropout, proj_dim=dimension,
                                                pad_id=0, use_stats_in_seq=use_stats)
        self.fuser = CrossModalFuser(
            dimension=dimension,
            n_heads=n_heads,
            dropout=dropout,
            use_stats=use_stats,
            droppath=0.0
        )
        self.with_projection = with_projection
        self.with_classification = with_classification
        if self.with_projection:
            self.proj_head = ProjectionHead(in_dim=dimension, hid_dim=dimension * 2, out_dim=128)
        if self.with_classification:
            self.cls_head = ClassificationHead(in_dim=dimension*3, hidden=dimension * 2, dropout=dropout)
            self.cls_head_sig = ClassificationHead(in_dim=dimension, hidden=dimension * 2, dropout=dropout)
            self.cls_head_seq = ClassificationHead(in_dim=dimension, hidden=dimension * 2, dropout=dropout)
    @torch.no_grad()
    def infer_shapes(self, T: int = 100, L: int = 13):
        B = 2
        sig = torch.randn(B, 13, T)
        seq_ids = torch.zeros(B, L, dtype=torch.long)
        stats = torch.randn(B, 13, 3)
        return self.forward(sig, seq_ids, stats)
    def forward(self, sig, seq_ids, stats=None):
        tokens_sig, g_sig = self.signal_encoder(sig, stats=stats)
        tokens_seq, g_seq = self.sequence_encoder(seq_ids, stats=stats)
        h_fused = self.fuser(tokens_seq, tokens_sig, stats=stats)  # [B,D]
        g_mix = 0.5 * (g_sig + g_seq)
        h = h_fused

        out = {
            "h": h,
            "g_sig": g_sig,
            "g_seq": g_seq,
            "z": None,
            "logits": None,
            "probs": None,
            "logits_sig": None,
            "logits_seq": None,
        }
        if self.with_projection:
            out["z"] = self.proj_head(h)
        if self.with_classification:

            h_main = torch.cat([h, g_sig, g_seq], dim=-1)  # [B, 3D]
            logits, probs = self.cls_head(h_main)

            logits_sig, _ = self.cls_head_sig(g_sig)
            logits_seq, _ = self.cls_head_seq(g_seq)

            out["logits"] = logits
            out["probs"] = probs
            out["logits_sig"] = logits_sig
            out["logits_seq"] = logits_seq
        return out

