import torch
import numpy as np
import torch.utils.data as Data
import csv
import random
import torch.nn.functional as F
from typing import Tuple, List
class SignalAugmenter:


    def __init__(self, level="weak", T=100, p_time_mask=0.6, p_perm=0.5, p_ch_drop=0.3):
        self.level = level
        self.T = T
        self.p_time_mask = p_time_mask
        self.p_perm = p_perm
        self.p_ch_drop = p_ch_drop


        if level == "weak":
            self.noise_std = 0.05
            self.shift_std = 0.10
            self.scale_low, self.scale_high = 0.90, 1.10
            self.dropout_pts = (2, 6)
            self.time_stretch = (0.95, 1.05)
            self.mask_span = (4, 8)
            self.permute_segments = (0, 0)

        else:
            self.noise_std = 0.08
            self.shift_std = 0.20
            self.scale_low, self.scale_high = 0.85, 1.15
            self.dropout_pts = (4, 12)
            self.time_stretch = (0.85, 1.15)
            self.mask_span = (8, 16)
            self.permute_segments = (3, 6)  # 把序列切成K段后小范围打乱

    @torch.no_grad()
    def _resample(self, x, stretch):

        C, T = x.shape
        target_T = self.T

        new_T = max(5, int(round(T * stretch)))
        x_ = x.unsqueeze(0)  # [1, C, T]
        x_ = F.interpolate(x_, size=new_T, mode="linear", align_corners=False)  # [1, C, new_T]
        x_ = F.interpolate(x_, size=target_T, mode="linear", align_corners=False)  # back to T
        return x_.squeeze(0)

    @torch.no_grad()
    def _time_mask(self, x, span):

        C, T = x.shape
        s = random.randint(*span)
        if s <= 0 or s >= T - 2:
            return x
        t0 = random.randint(1, T - s - 1)
        t1 = t0 + s

        left = x[:, t0-1:t0]
        right= x[:, t1:t1+1]
        fill = 0.5 * (left + right)
        x[:, t0:t1] = fill

        if self.level == "strong" and random.random() < 0.5:
            return self._time_mask(x, span)
        return x

    @torch.no_grad()
    def _permute_segments(self, x, k_range):

        C, T = x.shape
        k = random.randint(*k_range)
        if k <= 1:
            return x
        seg_len = T // k
        idxs = list(range(k))
        random.shuffle(idxs)
        out = []
        for i in idxs:
            s = i * seg_len
            e = T if i == k-1 else (i+1) * seg_len
            out.append(x[:, s:e])
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def _point_smooth(self, x, k_low_high):

        C, T = x.shape
        k = random.randint(*k_low_high)
        if T <= 2 or k <= 0:
            return x
        idx = torch.randint(1, T-1, (k,))
        for t in idx.tolist():
            x[:, t] = 0.5 * (x[:, t-1] + x[:, t+1])
        return x

    @torch.no_grad()
    def __call__(self, sig):

        x = sig.clone()

        C, T = x.shape

        scale = torch.empty(C, 1, dtype=x.dtype, device=x.device).uniform_(self.scale_low, self.scale_high)
        shift = torch.empty(C, 1, dtype=x.dtype, device=x.device).normal_(0.0, self.shift_std)
        x = x * scale + shift


        noise = torch.randn_like(x) * (self.noise_std * x.std(dim=1, keepdim=True))
        x = x + noise

        if random.random() < 0.8:
            stretch = random.uniform(*self.time_stretch)
            x = self._resample(x, stretch)


        if random.random() < self.p_time_mask:
            x = self._time_mask(x, self.mask_span)

        if self.level == "strong" and random.random() < self.p_perm:
            x = self._permute_segments(x, self.permute_segments)


        x = self._point_smooth(x, self.dropout_pts)

        if self.level == "strong" and random.random() < self.p_ch_drop:
            drop_idx = random.randrange(C)

            xa = x[drop_idx]
            kernel = torch.tensor([0.25, 0.5, 0.25], device=x.device, dtype=x.dtype).view(1,1,-1)
            xa_ = F.conv1d(xa.view(1,1,-1), kernel, padding=1).view(-1)
            x[drop_idx] = xa_

        return x

def z_score_normalize(data_list):

    data = np.array(data_list, dtype=np.float32)
    mean = np.mean(data)
    std = np.std(data) + 1e-8
    return (data - mean) / std


def safe_float(x, default=0.0):

    try:
        return float(x)
    except Exception:
        return default
VOCAB = {c:i for i, c in enumerate(
    ["N","A","C","G","T"]
)}
NUM_SIGNALS = 13
def encode_seq_13mer(seq_str):
    ids = [VOCAB.get(ch.upper(), VOCAB["N"]) for ch in seq_str[:NUM_SIGNALS]]
    if len(ids) < NUM_SIGNALS: ids += [VOCAB["N"]] * (NUM_SIGNALS - len(ids))
    return ids

@torch.no_grad()
def augment_sig(sig, noise_std=0.05, shift_std=0.1, scale_low=0.9, scale_high=1.1, dropout_pts=(2,6)):
    # sig: [5,100] float32
    x = sig.clone()
    ch = x.shape[0]
    scale = torch.empty(ch, 1, dtype=x.dtype, device=x.device).uniform_(scale_low, scale_high)
    shift = torch.empty(ch, 1, dtype=x.dtype, device=x.device).normal_(0.0, shift_std)
    x = x * scale + shift
    x = x + torch.randn_like(x) * noise_std
    k = random.randint(*dropout_pts)
    if x.shape[1] > 2 and k > 0:
        idx = torch.randint(1, x.shape[1]-1, (k,)).tolist()
        for t in idx:
            x[:, t] = 0.5 * (x[:, t-1] + x[:, t+1])
    return x

@torch.no_grad()
def recompute_stats_from_sig(sig, orig_len):
    # sig: [13, T]
    mean = sig.mean(dim=-1)
    std  = sig.std(dim=-1, unbiased=False)
    return torch.stack([mean, std, orig_len], dim=-1)  # [13,3]

@torch.no_grad()
def collate_supcon_13mer_train(samples):

    seq_once, sig_weak, sig_strong, sta_weak, sta_strong, labels = [], [], [], [], [], []
    strong_aug = SignalAugmenter(level="strong", T=100)

    for s, n, y in samples:
        n  = torch.as_tensor(n, dtype=torch.float32)   # [13,103]
        st = n[:, :3]                                  # [13,3]
        sg = n[:, 3:]                                  # [13,100]

        L = st[:, 2]                                   # [13]

        # view-1: strong
        sg_strong = strong_aug(sg)
        sg_strong = torch.nan_to_num(sg_strong, nan=0.0, posinf=0.0, neginf=0.0)


        sta_w = recompute_stats_from_sig(sg,        L)
        sta_s = recompute_stats_from_sig(sg_strong, L)


        seq_once.append(torch.tensor(encode_seq_13mer(s), dtype=torch.int64))
        sig_weak.append(sg);        sta_weak.append(sta_w)
        sig_strong.append(sg_strong); sta_strong.append(sta_s)
        labels.append(torch.tensor(int(y), dtype=torch.int64))


    seq_once   = torch.stack(seq_once,   0)  # [B,13]
    sig_weak_t   = torch.stack(sig_weak,   0)  # [B,13,100]
    sig_strong_t = torch.stack(sig_strong, 0)  # [B,13,100]
    sta_weak_t   = torch.stack(sta_weak,   0)  # [B,13,3]
    sta_strong_t = torch.stack(sta_strong, 0)  # [B,13,3]
    labels_t     = torch.stack(labels,     0)  # [B]


    seq   = torch.stack([seq_once,    seq_once],    0)  # [2,B,13]
    sig   = torch.stack([sig_weak_t,  sig_strong_t], 0) # [2,B,13,100]
    stats = torch.stack([sta_weak_t,  sta_strong_t], 0) # [2,B,13,3]
    return seq, sig, stats, labels_t


@torch.no_grad()
def collate_supcon_13mer_eval(samples):

    return collate_supcon_13mer_train(samples)

def load_dataset(filePath, feature_mode="both", mask=-1):

    sequence = []
    nano_data = []
    label = []

    with open(filePath, encoding='utf-8-sig') as f:
        reader = csv.reader(f, skipinitialspace=True)
        headers = next(reader)

        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 18:
                continue

            kmer = row[0].strip()
            if mask == 3:
                seq = 'N' * NUM_SIGNALS
            else:
                seq = kmer
            sequence.append(seq)

            try:
                signal_means = [safe_float(x) for x in row[1].split(",")]
                signal_stds = [safe_float(x) for x in row[2].split(",")]
                signal_lens = [safe_float(x) for x in row[3].split(",")]
            except Exception as e:
                continue


            # mask
            if mask == 0:
                signal_means = [0.0] * len(signal_means)
            elif mask == 1:
                signal_stds = [0.0] * len(signal_stds)
            elif mask == 2:
                signal_lens = [0.0] * len(signal_lens)


            raw_signals = []
            try:
                for i in range(4, 4 + NUM_SIGNALS):
                    if i >= len(row):
                        raise ValueError(f"Missing column {i + 1}")
                    sig_str = row[i]
                    sig_list = [safe_float(x) for x in sig_str.split(",") if x.strip()]
                    if len(sig_list) != 100:
                        print(
                            f"️ Warning: Row {row_idx}, Column {i + 1} signal length mismatch ({len(sig_list)}≠100), padding with zeros")
                        sig_list = sig_list[:100] + [0.0] * (100 - len(sig_list))
                    normalized_sig = z_score_normalize(sig_list).tolist()
                    raw_signals.append(normalized_sig)
            except Exception as e:
                print(f" Error: Row {row_idx} raw signal parsing failed: {e}, skipping this row")
                continue


            sample_features = []
            for pos in range(NUM_SIGNALS):
                if feature_mode == "stat":
                    feat = [
                        signal_means[pos],
                        signal_stds[pos],
                        signal_lens[pos]
                    ]
                elif feature_mode == "raw":
                    feat = raw_signals[pos]  # 100
                elif feature_mode == "both":
                    feat = [
                               signal_means[pos],
                               signal_stds[pos],
                               signal_lens[pos]
                           ] + raw_signals[pos]  # 3 + 100 = 103
                else:
                    raise ValueError(f"unknown feature_mode: {feature_mode}")

                sample_features.append(feat)

            nano_data.append(sample_features)

            try:
                lbl = int(float(row[17]))
                if lbl not in [0, 1]:
                    print(f"⚠ Warning: Row {row_idx} label mismatch ({lbl}), set to 0")
                    lbl = 0
                label.append(lbl)
            except Exception as e:
                print(f" Error: Row {row_idx} label parsing failed: {e}, set to 0")
                label.append(0)

    print(f"Successfully loaded {len(nano_data)} samples")
    return sequence, nano_data, label


def make_data(data_load):

    sequence, nano_data, label = data_load

    nano_tensor = torch.tensor(nano_data, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.long)
    return sequence, nano_tensor, label_tensor


class MyDataSet(Data.Dataset):

    def __init__(self, seq, nano, label):
        self.seq = seq  # List[str]
        self.nano = nano  # Tensor [N, 13, D]
        self.label = label  # Tensor [N]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.seq[idx], self.nano[idx], self.label[idx]



