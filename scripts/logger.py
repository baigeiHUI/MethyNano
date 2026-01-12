import os, csv, time
from collections import defaultdict
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val, k = 1):
        self.sum += float(val) * k
        self.n += int(k)

    @property
    def avg(self) :
        return self.sum / max(1, self.n)


class MetricTracker:

    def __init__(self):
        self.meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.meters[k].update(float(v), 1)

    def state_dict(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}

    def reset(self):
        self.meters.clear()



class CSVLogger:
    def __init__(self, log_dir: str, filename: str = "metrics.csv",fields: list[str] | None = None):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self._fh = open(self.path, "w", newline="")
        self._wr = None
        self._fields = fields
        if self._fields is not None:
            self._wr = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._wr.writeheader()
    def log_row(self, row: Dict[str, Any]):

        if self._fields is None:
            self._fields = list(row.keys())
            self._wr = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._wr.writeheader()
        filtered = {k: row.get(k, "") for k in self._fields}
        self._wr.writerow(filtered)
        self._fh.flush()
    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass



@torch.no_grad()
def contrastive_batch_metrics_pairwise(z1, z2, labels, t=2.0):

    B = z1.size(0)

    pos_sim = (z1 * z2).sum(dim=-1).mean().item()


    sim = z1 @ z2.t()  # [B, B]
    neq = (labels[:, None] != labels[None, :])
    negs = sim[neq]
    neg_sim = negs.mean().item() if negs.numel() else float('nan')

    pair_dist2 = (2 - 2 * (z1 * z2).sum(dim=-1)).clamp_min(0)  # [B]
    alignment = pair_dist2.mean().item()

    Z = torch.cat([z1, z2], 0).float()  # [2B, D]
    cos = Z @ Z.t()
    eye = torch.eye(2*B, device=Z.device, dtype=torch.bool)
    dist2_all = (2 - 2*cos).clamp_min(0)
    uniformity = torch.log(torch.exp(-t * dist2_all[~eye]).mean() + 1e-12).item()


    spread_all = dist2_all[~eye].mean().item()

    return dict(
        pos_sim=pos_sim,
        neg_sim=neg_sim,
        alignment=alignment,
        uniformity=uniformity,
        spread_all=spread_all,
        margin=(pos_sim - neg_sim)
    )


class ContrastiveLogger:
    def __init__(self, log_dir: str, use_tensorboard: bool = True, csv_name: str = "metrics.csv"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv = CSVLogger(log_dir, csv_name)
        self.tb: Optional[SummaryWriter] = SummaryWriter(log_dir) if (use_tensorboard and _HAS_TB) else None


    def log_step(
        self,
        step: int,
        lr: float,
        loss: Optional[float] = None,
        loss_con: Optional[float] = None,
        loss_cls: Optional[float] = None,
        supcon_stats: Optional[Dict[str, Any]] = None,
        batch_metrics: Optional[Dict[str, float]] = None,
        split: str = "train",
    ):
        if not self.tb:
            return
        self.tb.add_scalar(f"{split}/lr", lr, step)
        if loss is not None:
            self.tb.add_scalar(f"{split}/loss", float(loss), step)
        if loss_con is not None:
            self.tb.add_scalar(f"{split}/loss_con", float(loss_con), step)
        if loss_cls is not None:
            self.tb.add_scalar(f"{split}/loss_cls", float(loss_cls), step)
        if supcon_stats:
            for k, v in supcon_stats.items():
                self.tb.add_scalar(f"{split}/supcon/{k}", float(v), step)
        if batch_metrics:
            for k, v in batch_metrics.items():
                self.tb.add_scalar(f"{split}/con/{k}", float(v), step)


    def log_step_contrastive(
        self,
        step: int,
        loss: float,
        loss_con: float,
        loss_cls: float,
        lr: float,
        supcon_stats: Optional[Dict[str, Any]] = None,
        batch_metrics: Optional[Dict[str, float]] = None,
        split: str = "train",
    ):

        self.log_step(
            step=step,
            lr=lr,
            loss=None,
            loss_con=float(loss),
            loss_cls=None,
            supcon_stats=supcon_stats,
            batch_metrics=batch_metrics,
            split=split,
        )


    def log_epoch(self, epoch: int, split: str, meters: Dict[str, float]):
        row = {"time": time.time(), "epoch": epoch, "split": split}
        row.update(meters)
        self.csv.log_row(row)
        if self.tb:
            for k, v in meters.items():
                self.tb.add_scalar(f"{split}/epoch/{k}", float(v), epoch)

    def close(self):
        self.csv.close()
        if self.tb:
            self.tb.flush()
            self.tb.close()


@torch.no_grad()
def sigmoid_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:

    if logits.dim() == 2:

        probs = F.softmax(logits, dim=-1)[..., 1]
    else:
        probs = torch.sigmoid(logits)
    return probs


@torch.no_grad()
def binary_confusion_counts(
    probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, int]:

    preds = (probs >= threshold).to(torch.long)
    tp = int(((preds == 1) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())
    return dict(tp=tp, tn=tn, fp=fp, fn=fn)


@torch.no_grad()
def precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / max(1, (tp + fp))
    recall    = tp / max(1, (tp + fn))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f1=f1)


@torch.no_grad()
def auroc_auprc_binary(y_true: torch.Tensor, y_score: torch.Tensor) -> Dict[str, float]:

    y_true = y_true.to(torch.long)
    y_score = y_score.to(torch.float)

    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]

    P = y_true.sum().item()
    N = (y_true.numel() - P)

    tps = torch.cumsum(y_true, dim=0)
    fps = torch.cumsum(1 - y_true, dim=0)

    tps_ext = torch.cat([torch.zeros(1, device=tps.device, dtype=tps.dtype), tps])
    fps_ext = torch.cat([torch.zeros(1, device=fps.device, dtype=fps.dtype), fps])

    tpr = (tps_ext / max(1, P)).cpu()
    fpr = (fps_ext / max(1, N)).cpu()

    auroc = torch.trapz(tpr, fpr).item()

    precision = tps_ext / torch.clamp(tps_ext + fps_ext, min=1)
    recall    = tps_ext / max(1, P)

    precision = torch.cat([torch.ones(1), precision.cpu()])
    recall    = torch.cat([torch.zeros(1), recall.cpu()])

    auprc = torch.trapz(precision, recall).item()

    return dict(auroc=auroc, auprc=auprc)

class ClassificationAccumulator:

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_count = 0
        self.tp = self.fp = self.tn = self.fn = 0
        self.y_true = []
        self.y_score = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: Optional[torch.Tensor] = None):
        B = targets.numel()
        probs = sigmoid_probs_from_logits(logits)  # [B]

        cm = binary_confusion_counts(probs, targets, self.threshold)
        self.tp += cm["tp"]; self.fp += cm["fp"]; self.tn += cm["tn"]; self.fn += cm["fn"]

        if loss is not None:
            self.total_loss += float(loss.item()) * B
        self.total_count += B
        self.y_true.append(targets.detach().to(torch.long).cpu())
        self.y_score.append(probs.detach().cpu())

    def compute(self) -> Dict[str, float]:

        avg_loss = self.total_loss / max(1, self.total_count)
        acc = (self.tp + self.tn) / max(1, (self.tp + self.tn + self.fp + self.fn))
        prf = precision_recall_f1_from_counts(self.tp, self.fp, self.fn)

        # AUROC/AUPRC
        y_true = torch.cat(self.y_true, dim=0) if len(self.y_true) > 0 else torch.empty(0, dtype=torch.long)
        y_score = torch.cat(self.y_score, dim=0) if len(self.y_score) > 0 else torch.empty(0)
        if y_true.numel() > 0:
            au = auroc_auprc_binary(y_true, y_score)
        else:
            au = dict(auroc=float("nan"), auprc=float("nan"))

        out = dict(
            loss=avg_loss,
            acc=acc,
            precision=prf["precision"],
            recall=prf["recall"],
            f1=prf["f1"],
            auroc=au["auroc"],
            auprc=au["auprc"],
            tp=float(self.tp), fp=float(self.fp), tn=float(self.tn), fn=float(self.fn),
            n=float(self.total_count),
        )
        return out



class ClassificationLogger:

    def __init__(self, log_dir: str, use_tensorboard: bool = True, csv_name: str = "metrics_cls.csv"):
        from collections import defaultdict
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        cls_fields = [
            "time", "epoch", "split",
            "loss", "acc", "precision", "recall", "f1",
            "auroc", "auprc",
            "tp", "fp", "tn", "fn",
            "n",
        ]

        self.csv = CSVLogger(log_dir, csv_name, cls_fields)
        self.tb: Optional[SummaryWriter] = SummaryWriter(log_dir) if (use_tensorboard and _HAS_TB) else None

    @torch.no_grad()
    def _batch_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        probs = sigmoid_probs_from_logits(logits)
        cm = binary_confusion_counts(probs, targets)
        prf = precision_recall_f1_from_counts(cm["tp"], cm["fp"], cm["fn"])
        acc = (cm["tp"] + cm["tn"]) / max(1, (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]))

        au = auroc_auprc_binary(targets.to(torch.long), probs)
        return dict(acc=acc, precision=prf["precision"], recall=prf["recall"], f1=prf["f1"],
                    auroc=au["auroc"], auprc=au["auprc"])

    def log_step(self, step, split, loss, lr,
                 logits: torch.Tensor, targets: torch.Tensor):
        if not self.tb:
            return
        self.tb.add_scalar(f"{split}/loss", float(loss), step)
        self.tb.add_scalar(f"{split}/lr", float(lr), step)
        m = self._batch_metrics(logits.detach(), targets.detach())
        for k, v in m.items():
            self.tb.add_scalar(f"{split}/step/{k}", float(v), step)

    def log_epoch(self, epoch: int, split: str, meters: Dict[str, float]):

        row = {"time": time.time(), "epoch": epoch, "split": split}
        row.update(meters)
        self.csv.log_row(row)


        if self.tb:
            for k, v in meters.items():
                self.tb.add_scalar(f"{split}/epoch/{k}", float(v), epoch)

    def close(self):
        self.csv.close()
        if self.tb:
            self.tb.flush()
            self.tb.close()