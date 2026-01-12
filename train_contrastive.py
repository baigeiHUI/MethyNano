import os
import math
import random
import argparse
from typing import Optional, Dict, Tuple

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


from scripts.dataLoader import (
    load_dataset,
    make_data,
    MyDataSet,
    encode_seq_13mer,
collate_supcon_13mer_train,
 collate_supcon_13mer_eval
)
from scripts.loss import ContrastiveLoss
from scripts.logger import ContrastiveLogger, contrastive_batch_metrics_pairwise
from moduls import MethyNano



def _finite(*tensors) -> bool:

    for t in tensors:
        if isinstance(t, torch.Tensor) and (not torch.isfinite(t).all().item()):
            return False
    return True


class WarmupCosine:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int,
                 min_lr: float = 1e-5, base_lr: float = 1e-3):
        self.opt = optimizer
        self.t_total = total_steps
        self.t_warm = max(1, warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lr = float(base_lr)
        self.step_id = 0

    def step(self) -> float:
        self.step_id += 1
        t = self.step_id
        if t <= self.t_warm:
            lr = self.base_lr * t / self.t_warm
        else:
            progress = (t - self.t_warm) / max(1, self.t_total - self.t_warm)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr

def _epoch_metrics_from_logits(logits_cat: torch.Tensor, labels_cat: torch.Tensor) -> Dict[str, float]:
    """
    logits_cat: [N,2], labels_cat: [N]
    return: dict(loss, acc, precision, recall, f1, auroc, auprc)
    """
    ce = F.cross_entropy(logits_cat, labels_cat).item()

    probs = F.softmax(logits_cat, dim=-1)[:, 1].detach().cpu().numpy()
    ys = labels_cat.detach().cpu().numpy().astype("int64")
    preds = (probs >= 0.5).astype("int64")

    tp = int(((preds == 1) & (ys == 1)).sum())
    tn = int(((preds == 0) & (ys == 0)).sum())
    fp = int(((preds == 1) & (ys == 0)).sum())
    fn = int(((preds == 0) & (ys == 1)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    # AUROC / AUPRC
    order = probs.argsort()[::-1]
    y_sorted = ys[order]
    P = max(1, (ys == 1).sum())
    N = max(1, (ys == 0).sum())
    tp_c = (y_sorted == 1).cumsum()
    fp_c = (y_sorted == 0).cumsum()
    tpr = tp_c / P
    fpr = fp_c / N
    auroc = float(np.trapz(np.concatenate([[0.0], tpr, [1.0]]),
                           np.concatenate([[0.0], fpr, [1.0]])))
    prec_curve = tp_c / np.maximum(1, tp_c + fp_c)
    rec_curve = tp_c / P
    auprc = float(np.trapz(np.concatenate([[1.0], prec_curve,
                                           [prec_curve[-1] if prec_curve.size else 1.0]]),
                           np.concatenate([[0.0], rec_curve, [1.0]])))
    return dict(loss=ce, acc=acc, precision=precision,
                recall=recall, f1=f1, auroc=auroc, auprc=auprc)


@torch.inference_mode()
def evaluate_joint(
    model: nn.Module,
    val_loader,
    device: torch.device,
    temperature: float = 0.1,
    t_uniform: float = 2.0,
) -> Dict[str, float]:

    supcon_eval = ContrastiveLoss(
        temperature=temperature,
        class_balance=False,
        assert_unit_norm=False,
        weak_same_alpha=0.01,
    )
    model.eval()

    all_logits, all_labels = [], []
    tot_cls_loss, n_cls = 0.0, 0

    agg_sum = {"alignment": 0.0, "uniformity": 0.0,
               "pos_sim": 0.0, "neg_sim": 0.0}
    agg_cnt = 0
    tot_con_loss = 0.0

    pbar = tqdm(total=len(val_loader), desc="Valid (joint)", leave=False, dynamic_ncols=True)
    for seq, sig, stats, labels in val_loader:
        seq1, seq2 = seq[0].to(device), seq[1].to(device)
        sig1, sig2 = sig[0].to(device), sig[1].to(device)
        sta1, sta2 = stats[0].to(device), stats[1].to(device)
        y = labels.to(device)
        bsz = y.size(0)

        out1 = model(sig1, seq1, sta1)
        logits = out1["logits"]
        cls_loss = F.cross_entropy(logits, y)

        cls_loss = cls_loss * 5.0

        tot_cls_loss += float(cls_loss.item()) * bsz
        n_cls += bsz
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

        z1 = F.normalize(out1["z"].float(), dim=-1, eps=1e-6)
        out2 = model(sig2, seq2, sta2)
        z2 = F.normalize(out2["z"].float(), dim=-1, eps=1e-6)

        con_loss = supcon_eval(z1, z2, y)
        tot_con_loss += float(con_loss.item()) * bsz

        m = contrastive_batch_metrics_pairwise(z1, z2, y, t=t_uniform)
        for k in agg_sum.keys():
            if k in m and m[k] == m[k]:  # 过滤 NaN
                agg_sum[k] += float(m[k]) * bsz
        agg_cnt += bsz

        pbar.set_postfix({
            "v_cls": f"{cls_loss.item():.4f}",
            "v_con": f"{con_loss.item():.4f}",
            "pos": f"{m['pos_sim']:.3f}",
            "neg": f"{m['neg_sim']:.3f}",
        })
        pbar.update(1)
    pbar.close()

    if n_cls == 0:
        cls_m = dict(loss=float("nan"), acc=float("nan"), precision=float("nan"),
                     recall=float("nan"), f1=float("nan"),
                     auroc=float("nan"), auprc=float("nan"))
    else:
        logits_cat = torch.cat(all_logits, 0)
        labels_cat = torch.cat(all_labels, 0)
        cls_m = _epoch_metrics_from_logits(logits_cat, labels_cat)
        cls_m["loss"] = tot_cls_loss / n_cls

    if agg_cnt == 0:
        con_loss_avg = float("nan")
        alignment = uniformity = pos_sim = neg_sim = float("nan")
    else:
        con_loss_avg = tot_con_loss / agg_cnt
        alignment = agg_sum["alignment"] / agg_cnt
        uniformity = agg_sum["uniformity"] / agg_cnt
        pos_sim = agg_sum["pos_sim"] / agg_cnt
        neg_sim = agg_sum["neg_sim"] / agg_cnt

    return dict(
        val_loss_cls=cls_m["loss"],
        auroc=cls_m["auroc"],
        auprc=cls_m["auprc"],
        acc=cls_m["acc"],
        precision=cls_m["precision"],
        recall=cls_m["recall"],
        f1=cls_m["f1"],
        val_loss_con=con_loss_avg,
        val_alignment=alignment,
        val_uniformity=uniformity,
        val_pos_sim=pos_sim,
        val_neg_sim=neg_sim,
    )

def train_joint(
    model: nn.Module,
    train_loader,
    val_loader,
    logger: Optional[ContrastiveLogger],
    epochs: int = 50,
    base_lr: float = 1e-3,
    min_lr: float = 1e-4,
    weight_decay: float = 1e-5,
    warmup_ratio: float = 0.05,
    temperature: float = 0.1,
    grad_clip: float = 1.0,
    amp: bool = False,
    ckpt_dir: str = "./models/pretrain",
    device: Optional[torch.device] = None,
) -> Dict[str, any]:

    os.makedirs(ckpt_dir, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    sched = WarmupCosine(optimizer, total_steps, warmup_steps,
                         min_lr=min_lr, base_lr=base_lr)

    supcon = ContrastiveLoss(
        temperature=temperature,
        class_balance=False,
        assert_unit_norm=False,
        weak_same_alpha=0.01,
    )

    ema_decay = 0.999
    ema_model = MethyNano(
        with_projection=True,
        with_classification=True,
        dimension=256,
        n_heads=8,
        dropout=0.1,
        base_sig=160,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    def _ema_update():
        with torch.no_grad():
            msd = model.state_dict()
            esd = ema_model.state_dict()
            for k in esd.keys():
                v_model = msd[k]
                v_ema = esd[k]
                if not v_ema.dtype.is_floating_point:
                    v_ema.copy_(v_model)
                else:
                    v_ema.mul_(ema_decay).add_(v_model, alpha=1.0 - ema_decay)

    scaler = GradScaler(enabled=amp)

    best_metric = -float("inf")
    best_path = None
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        sum_con, sum_cls, n_samp = 0.0, 0.0, 0
        train_logits_all, train_labels_all = [], []

        train_sum = {"alignment": 0.0, "uniformity": 0.0,
                     "pos_sim": 0.0, "neg_sim": 0.0}
        train_cnt = 0

        pbar = tqdm(total=len(train_loader),
                    desc=f"Epoch {epoch}/{epochs}", leave=False, dynamic_ncols=True)

        for seq, sig, stats, labels in train_loader:
            seq1, seq2 = seq[0].to(device), seq[1].to(device)
            sig1, sig2 = sig[0].to(device), sig[1].to(device)
            sta1, sta2 = stats[0].to(device), stats[1].to(device)
            y = labels.to(device)
            bsz = y.size(0)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):

                out1 = model(sig1, seq1, sta1)
                logits = out1["logits"]
                loss_cls = F.cross_entropy(logits, y, label_smoothing=0.05)
                loss_cls = loss_cls * 5.0

                z1 = F.normalize(out1["z"].float(), dim=-1, eps=1e-6)
                out2 = model(sig2, seq2, sta2)
                z2 = F.normalize(out2["z"].float(), dim=-1, eps=1e-6)

                loss_con = supcon(z1, z2, y)
                loss = loss_cls + loss_con

            if not _finite(loss):
                raise RuntimeError("loss is NaN/Inf")

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr = sched.step()

            sum_con += float(loss_con.item()) * bsz
            sum_cls += float(loss_cls.item()) * bsz
            n_samp += bsz

            train_logits_all.append(logits.detach().cpu())
            train_labels_all.append(y.detach().cpu())

            with torch.no_grad():
                m = contrastive_batch_metrics_pairwise(z1, z2, y, t=2.0)
                for k in train_sum.keys():
                    train_sum[k] += float(m[k]) * bsz
                train_cnt += bsz

            pbar.set_postfix({
                "L": f"{loss.item():.4f}",
                "L_cls": f"{loss_cls.item():.4f}",
                "L_con": f"{loss_con.item():.4f}",
                "lr": f"{lr:.2e}",
                "pos": f"{m['pos_sim']:.3f}",
                "neg": f"{m['neg_sim']:.3f}",
                "ali": f"{m['alignment']:.3f}",
            })
            pbar.update(1)

            _ema_update()

            if logger is not None:
                logger.log_step_contrastive(
                    step=global_step,
                    lr=float(lr),
                    loss=float(loss.item()),
                    loss_con=float(loss_con.item()),
                    loss_cls=float(loss_cls.item()),
                    supcon_stats=getattr(supcon, "_last_stats", None),
                    batch_metrics=m,
                    split="train",
                )
            global_step += 1
        pbar.close()

        if len(train_logits_all) > 0:
            train_logits_cat = torch.cat(train_logits_all, 0)
            train_labels_cat = torch.cat(train_labels_all, 0)
            train_cls_m = _epoch_metrics_from_logits(train_logits_cat, train_labels_cat)
        else:
            train_cls_m = dict(loss=float("nan"), acc=float("nan"), precision=float("nan"),
                               recall=float("nan"), f1=float("nan"),
                               auroc=float("nan"), auprc=float("nan"))

        train_metrics_epoch = {
            "train_loss_con": sum_con / max(1, n_samp),
            "train_loss_cls": sum_cls / max(1, n_samp),
            **{f"train_{k}": (train_sum[k] / max(1, train_cnt)) for k in train_sum.keys()},
            "train_acc": train_cls_m["acc"],
            "train_precision": train_cls_m["precision"],
            "train_recall": train_cls_m["recall"],
            "train_f1": train_cls_m["f1"],
            "train_auroc": train_cls_m["auroc"],
            "train_auprc": train_cls_m["auprc"],
        }

        val_m = evaluate_joint(
            ema_model, val_loader,
            device=device,
            temperature=temperature,
            t_uniform=2.0,
        )

        if logger is not None:
            meters = {**train_metrics_epoch,
                      **{f"val_{k}": v for k, v in val_m.items()}}
            logger.log_epoch(epoch, split="joint", meters=meters)

        val_score = float(val_m.get("auroc", float("-inf")))
        if math.isfinite(val_score) and val_score > best_metric:
            if best_path and os.path.isfile(best_path):
                try:
                    os.remove(best_path)
                except Exception:
                    pass
            best_metric = val_score
            best_path = os.path.join(
                ckpt_dir, f"joint_best_auroc={best_metric:.6f}_epoch{epoch:03d}.pth"
            )
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "auroc": best_metric}, best_path)
            print(f"[CKPT] New best AUROC {best_metric:.6f} -> {best_path}")

        print(
            f"Epoch {epoch:03d} | "
            f"train L_cls={train_metrics_epoch['train_loss_cls']:.4f} "
            f"L_con={train_metrics_epoch['train_loss_con']:.4f} | "
            f"train ACC={train_cls_m['acc']:.4f} F1={train_cls_m['f1']:.4f} "
            f"AUROC={train_cls_m['auroc']:.4f} AUPRC={train_cls_m['auprc']:.4f} | "
            f"val AUROC={val_m['auroc']:.4f} AUPRC={val_m['auprc']:.4f} "
            f"ACC={val_m['acc']:.4f} F1={val_m['f1']:.4f} | "
            f"val_cls={val_m['val_loss_cls']:.4f} val_con={val_m['val_loss_con']:.4f} | "
            f"best_auroc={best_metric:.6f}"
        )

    return {"best_auroc": best_metric, "best_path": best_path}


def build_dataloaders(
    train_csv: str,
    val_csv: str,
    batch_size: int = 512,
    num_workers: int = 8,
):
    trainData = load_dataset(train_csv, feature_mode="both", mask=-1)
    valData = load_dataset(val_csv, feature_mode="both", mask=-1)

    trn_seq, trn_nano, trn_label = make_data(trainData)
    val_seq, val_nano, val_label = make_data(valData)

    train_ds = MyDataSet(trn_seq, trn_nano, trn_label)
    val_ds = MyDataSet(val_seq, val_nano, val_label)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_supcon_13mer_train,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_supcon_13mer_eval,
        drop_last=False,
    )
    return train_loader, val_loader


# ======================
# CLI
# ======================
def get_args():
    p = argparse.ArgumentParser("Contrastive pretraining (SupCon + cls) for MethyNano 13-mer")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logdir", type=str, default="./runs/pretrain_supcon")
    p.add_argument("--ckpt_dir", type=str, default="./models/pretrain_supcon")
    p.add_argument("--resume", type=str, default=None,
                   help="optional: resume model weights from .pth (key 'model')")
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    train_loader, val_loader = build_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = MethyNano(
        with_projection=True,
        with_classification=True,
        dimension=256,
        n_heads=8,
        dropout=0.1,
        base_sig=160,
    )
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[RESUME] loaded: {args.resume} | missing={len(missing)} unexpected={len(unexpected)}")

    logger = ContrastiveLogger(args.logdir, use_tensorboard=True)

    stats = train_joint(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        temperature=args.tau,
        grad_clip=1.0,
        amp=True,
        ckpt_dir=args.ckpt_dir,
        device=device,
    )
    print("Best:", stats)
    logger.close()


if __name__ == "__main__":
    main()