# finetune_cls_revised.py
import os
import math
import argparse
import random
import copy
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.dataLoader import load_dataset, make_data, MyDataSet, encode_seq_13mer
from scripts.logger import CSVLogger
from moduls import MethyNano


SIG_SCALAR_MODE = os.getenv("SIG_SCALAR_MODE", "none")


def seed_everything(seed: int = 12, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def _nz(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _collate_cls_13mer(batch):
    """
    batch: List[(seq_str, nano[13,103], label)]
    return:
      seq_ids [B,13] (long)
      sig     [B,13,100] (float32)
      stats   [B,13,3]   (float32)
      labels  [B]        (long)
    """
    seq_ids, sig, stats, labels = [], [], [], []
    for s, n, y in batch:
        n = torch.as_tensor(n, dtype=torch.float32)
        st = n[:, :3]
        sg = n[:, 3:]

        if SIG_SCALAR_MODE != "none":
            if SIG_SCALAR_MODE == "first":
                v = sg[:, 0]
            elif SIG_SCALAR_MODE == "center":
                v = sg[:, 50]
            elif SIG_SCALAR_MODE == "mean":
                v = sg.mean(dim=-1)
            else:
                v = sg.mean(dim=-1)
            sg = v.unsqueeze(-1).expand(-1, sg.size(-1))

        seq_ids.append(torch.tensor(encode_seq_13mer(s), dtype=torch.long))
        stats.append(_nz(st))
        sig.append(_nz(sg))
        labels.append(torch.tensor(int(y), dtype=torch.long))

    return (
        torch.stack(seq_ids, 0),
        torch.stack(sig, 0),
        torch.stack(stats, 0),
        torch.stack(labels, 0),
    )


def build_cls_dataloaders(train_csv, val_csv, batch_size=256, num_workers=8):
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
        collate_fn=_collate_cls_13mer,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=_collate_cls_13mer,
        drop_last=False,
    )

    trn_label_np = np.asarray(trn_label, dtype=np.int64)
    val_label_np = np.asarray(val_label, dtype=np.int64)
    train_counts = np.bincount(trn_label_np, minlength=2)
    val_counts = np.bincount(val_label_np, minlength=2)
    return train_loader, val_loader, train_counts, val_counts


def _find_best_threshold(probs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    cand = np.linspace(0.10, 0.90, 81)
    best_thr, best_f1 = 0.5, -1.0
    for thr in cand:
        preds = (probs >= thr).astype("int32")
        tp = int(((preds == 1) & (ys == 1)).sum())
        fp = int(((preds == 1) & (ys == 0)).sum())
        fn = int(((preds == 0) & (ys == 1)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


def _epoch_metrics_from_logits(
    logits_cat: torch.Tensor,
    labels_cat: torch.Tensor,
    threshold: float = 0.5,
    search_best_threshold: bool = False,
) -> Dict[str, float]:
    ce = F.cross_entropy(logits_cat, labels_cat).item()

    probs = F.softmax(logits_cat, dim=-1)[:, 1].detach().cpu().numpy()
    ys = labels_cat.detach().cpu().numpy().astype("int32")
    preds = (probs >= threshold).astype("int32")

    tp = int(((preds == 1) & (ys == 1)).sum())
    tn = int(((preds == 0) & (ys == 0)).sum())
    fp = int(((preds == 1) & (ys == 0)).sum())
    fn = int(((preds == 0) & (ys == 1)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    order = probs.argsort()[::-1]
    y_sorted = ys[order]
    P = max(1, int((ys == 1).sum()))
    N = max(1, int((ys == 0).sum()))
    tp_c = (y_sorted == 1).cumsum()
    fp_c = (y_sorted == 0).cumsum()
    tpr = tp_c / P
    fpr = fp_c / N
    auroc = float(
        np.trapz(
            np.concatenate([[0.0], tpr, [1.0]]),
            np.concatenate([[0.0], fpr, [1.0]]),
        )
    )

    prec_curve = tp_c / np.maximum(1, tp_c + fp_c)
    rec_curve = tp_c / P
    auprc = float(
        np.trapz(
            np.concatenate([[1.0], prec_curve, [prec_curve[-1] if prec_curve.size else 1.0]]),
            np.concatenate([[0.0], rec_curve, [1.0]]),
        )
    )

    out = {
        "loss": ce,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "threshold": float(threshold),
    }
    if search_best_threshold:
        best_thr, best_f1 = _find_best_threshold(probs, ys)
        out["best_thr"] = best_thr
        out["best_f1"] = best_f1
    return out


@torch.no_grad()
def run_eval(model, val_loader, device, threshold=0.5, search_best_threshold=True):
    model.eval()
    all_logits, all_labels = [], []
    pbar = tqdm(total=len(val_loader), desc="Valid", leave=False, dynamic_ncols=True)
    for seq_ids, sig, stats, labels in val_loader:
        seq_ids = seq_ids.to(device, non_blocking=True)
        sig = _nz(sig).to(device, non_blocking=True)
        stats = _nz(stats).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(sig, seq_ids, stats)
        logits = out["logits"]
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        pbar.update(1)
    pbar.close()
    logits_cat = torch.cat(all_logits, 0)
    labels_cat = torch.cat(all_labels, 0)
    return _epoch_metrics_from_logits(
        logits_cat, labels_cat, threshold=threshold, search_best_threshold=search_best_threshold
    )


def make_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float):
    def lr_lambda(epoch_idx: int):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            start = 0.1
            return start + (1.0 - start) * ((epoch_idx + 1) / warmup_epochs)

        if total_epochs <= warmup_epochs:
            return 1.0
        t = epoch_idx - warmup_epochs
        T = total_epochs - warmup_epochs
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / T))
        return (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_train(
    model,
    train_loader,
    val_loader,
    train_counts,
    val_counts,
    device,
    epochs=25,
    lr=1e-3,
    min_lr=8e-4,
    warmup_epochs=5,
    ce_only_epochs=3,
    weight_decay=1e-5,
    ckpt_dir="./checkpoints/finetune",
    logdir="./runs/finetune",
    tag="finetune",
    amp=False,
    grad_clip=1.0,
    monitor="auprc",
    alpha_kl=2.0,
    aux_weight=0.1,
    threshold=0.5,
    use_ema=True,
    ema_decay=0.999,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    for p in model.parameters():
        p.requires_grad_(True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and ("bias" not in n) and ("bn" not in n) and ("ln" not in n) and ("norm" not in n):
            decay.append(p)
        else:
            no_decay.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    scheduler = make_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        base_lr=lr,
        min_lr=min_lr,
    )

    neg_n = max(1, int(train_counts[0]))
    pos_n = max(1, int(train_counts[1]))
    cls_weights = torch.tensor([1.0, neg_n / pos_n], dtype=torch.float32, device=device)
    if cls_weights[1] > 10.0:
        cls_weights[1] = 10.0

    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

        def _update_ema(ema_m, model_m, decay: float):
            msd = model_m.state_dict()
            esd = ema_m.state_dict()
            for k, v in esd.items():
                v_model = msd[k]
                if not v.dtype.is_floating_point:
                    v.copy_(v_model)
                else:
                    v.mul_(decay).add_(v_model, alpha=1.0 - decay)

    fields = [
        "epoch", "lr", "kl_w",
        "train_loss", "train_acc", "train_f1", "train_auroc", "train_auprc", "train_precision", "train_recall",
        "val_loss", "val_acc", "val_f1", "val_auroc", "val_auprc", "val_precision", "val_recall",
        "val_best_thr", "val_best_f1",
    ]
    csv = CSVLogger(log_dir=logdir, filename="metrics.csv", fields=fields)

    print(f"[DATA] train neg/pos = {neg_n}/{pos_n} | val neg/pos = {int(val_counts[0])}/{int(val_counts[1])}")
    print(f"[LOSS] class_weights = {cls_weights.detach().cpu().tolist()}")
    print(f"[SCHED] warmup_epochs={warmup_epochs}, ce_only_epochs={ce_only_epochs}, lr={lr}, min_lr={min_lr}")
    print(f"[ENV] SIG_SCALAR_MODE={SIG_SCALAR_MODE}")

    best_val = -1.0
    best_path = None
    last_path = None

    for ep in range(1, epochs + 1):
        model.train()
        cur_lr = optimizer.param_groups[0]["lr"]

        if ep <= ce_only_epochs:
            kl_w = 0.0
        else:
            ramp_span = max(1, warmup_epochs)
            kl_w = alpha_kl * min(1.0, (ep - ce_only_epochs) / ramp_span)

        step_bar = tqdm(total=len(train_loader), desc=f"Train {ep}/{epochs}", leave=False, dynamic_ncols=True)
        all_logits, all_labels = [], []

        for seq_ids, sig, stats, labels in train_loader:
            seq_ids = seq_ids.to(device, non_blocking=True)
            sig = _nz(sig).to(device, non_blocking=True)
            stats = _nz(stats).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                if ep <= ce_only_epochs:
                    out = model(sig, seq_ids, stats)
                    logits = out["logits"]
                    loss_ce = F.cross_entropy(
                        logits,
                        labels,
                        weight=cls_weights,
                        label_smoothing=0.0,
                    )

                    loss_aux = 0.0
                    if out.get("logits_sig") is not None:
                        loss_aux = loss_aux + F.cross_entropy(out["logits_sig"], labels, weight=cls_weights)
                    if out.get("logits_seq") is not None:
                        loss_aux = loss_aux + F.cross_entropy(out["logits_seq"], labels, weight=cls_weights)

                    loss = loss_ce + aux_weight * loss_aux
                else:
                    out1 = model(sig, seq_ids, stats)
                    out2 = model(sig, seq_ids, stats)
                    logits1 = out1["logits"]
                    logits2 = out2["logits"]

                    loss_ce1 = F.cross_entropy(
                        logits1,
                        labels,
                        weight=cls_weights,
                        label_smoothing=0.01,
                    )
                    loss_ce2 = F.cross_entropy(
                        logits2,
                        labels,
                        weight=cls_weights,
                        label_smoothing=0.01,
                    )
                    loss_ce = 0.5 * (loss_ce1 + loss_ce2)

                    p1 = F.log_softmax(logits1, dim=-1)
                    p2 = F.log_softmax(logits2, dim=-1)
                    loss_kl = 0.5 * (
                        F.kl_div(p1, p2.exp(), reduction="batchmean") +
                        F.kl_div(p2, p1.exp(), reduction="batchmean")
                    )

                    loss_aux = 0.0
                    if out1.get("logits_sig") is not None:
                        loss_aux = loss_aux + F.cross_entropy(out1["logits_sig"], labels, weight=cls_weights)
                    if out1.get("logits_seq") is not None:
                        loss_aux = loss_aux + F.cross_entropy(out1["logits_seq"], labels, weight=cls_weights)

                    loss = loss_ce + kl_w * loss_kl + aux_weight * loss_aux
                    logits = 0.5 * (logits1 + logits2)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if use_ema and ema_model is not None:
                _update_ema(ema_model, model, decay=ema_decay)

            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)[:, 1]
                pred = (probs >= threshold).long()
                acc_b = (pred == labels).float().mean().item()
                pos_rate = pred.float().mean().item()
            step_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{acc_b:.3f}",
                    "pos": f"{pos_rate:.3f}",
                    "klw": f"{kl_w:.2f}",
                }
            )
            step_bar.update(1)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        step_bar.close()

        tr_logits = torch.cat(all_logits, 0)
        tr_labels = torch.cat(all_labels, 0)
        train_m = _epoch_metrics_from_logits(tr_logits, tr_labels, threshold=threshold, search_best_threshold=False)

        eval_model = ema_model if (use_ema and ema_model is not None) else model
        val_m = run_eval(eval_model, val_loader, device=device, threshold=threshold, search_best_threshold=True)
        scheduler.step()

        row = {
            "epoch": ep,
            "lr": cur_lr,
            "kl_w": kl_w,
            "train_loss": train_m["loss"],
            "train_acc": train_m["acc"],
            "train_f1": train_m["f1"],
            "train_auroc": train_m["auroc"],
            "train_auprc": train_m["auprc"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["acc"],
            "val_f1": val_m["f1"],
            "val_auroc": val_m["auroc"],
            "val_auprc": val_m["auprc"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_best_thr": val_m.get("best_thr", threshold),
            "val_best_f1": val_m.get("best_f1", val_m["f1"]),
        }
        csv.log_row(row)

        print(
            f"Epoch {ep:03d} | lr={cur_lr:.2e} kl_w={kl_w:.2f} | "
            f"train: loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} f1={train_m['f1']:.3f} "
            f"auroc={train_m['auroc']:.3f} auprc={train_m['auprc']:.3f} | "
            f"val: loss={val_m['loss']:.4f} acc={val_m['acc']:.3f} f1={val_m['f1']:.3f} "
            f"best_f1={val_m.get('best_f1', val_m['f1']):.3f} best_thr={val_m.get('best_thr', threshold):.2f} "
            f"auroc={val_m['auroc']:.3f} auprc={val_m['auprc']:.3f}"
        )

        new_last = os.path.join(
            ckpt_dir,
            f"{tag}_last_ep{ep:03d}_f1={val_m['f1']:.4f}_acc={val_m['acc']:.4f}_auc={val_m['auprc']:.4f}.pth",
        )
        torch.save({"model": eval_model.state_dict(), "epoch": ep, "metrics": {"train": train_m, "val": val_m}}, new_last)
        if last_path and os.path.isfile(last_path) and last_path != new_last:
            try:
                os.remove(last_path)
            except Exception:
                pass
        last_path = new_last

        score = val_m.get(monitor, float("nan"))
        if math.isfinite(score) and score > best_val:
            if best_path and os.path.isfile(best_path):
                try:
                    os.remove(best_path)
                except Exception:
                    pass
            best_val = score
            best_path = os.path.join(
                ckpt_dir,
                f"{tag}_best_ep{ep:03d}_f1={val_m['f1']:.4f}_acc={val_m['acc']:.4f}_auc={val_m['auprc']:.4f}.pth",
            )
            torch.save({"model": eval_model.state_dict(), "epoch": ep, "metrics": {"train": train_m, "val": val_m}}, best_path)
            print(f"[CKPT] New BEST -> {best_path}")

    csv.close()
    return {"best_path": best_path, "last_path": last_path}


def get_args():
    p = argparse.ArgumentParser("Global fine-tuning for MethyNano (classification)")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=9e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--ce_only_epochs", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--alpha_kl", type=float, default=2.0)
    p.add_argument("--aux_weight", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints/finetune")
    p.add_argument("--logdir", type=str, default="./runs/finetune")
    p.add_argument("--resume", type=str, default=None, help="path to pretrain ckpt (.pth) saved by contrastive stage")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.devices > 0 else "cpu")

    is_finetune = args.resume is not None
    mode_tag = "finetune" if is_finetune else "cls"
    print(f"[MODE] {'Fine-tune from pretrained' if is_finetune else 'Train classification from scratch'} | tag={mode_tag}")

    train_loader, val_loader, train_counts, val_counts = build_cls_dataloaders(
        args.train_csv, args.val_csv, batch_size=args.batch_size, num_workers=args.workers
    )

    model = MethyNano(
        with_projection=False,
        with_classification=True,
        dimension=256,
        n_heads=8,
        dropout=0.1,
        base_sig=160,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[RESUME] loaded: {args.resume} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[RESUME] none -> training from random init (classification)")

    out = run_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_counts=train_counts,
        val_counts=val_counts,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        ce_only_epochs=args.ce_only_epochs,
        weight_decay=args.weight_decay,
        ckpt_dir=args.ckpt_dir,
        logdir=args.logdir,
        tag=mode_tag,
        amp=args.amp,
        grad_clip=1.0,
        monitor="auprc",
        alpha_kl=args.alpha_kl,
        aux_weight=args.aux_weight,
        threshold=args.threshold,
        use_ema=(not args.no_ema),
        ema_decay=0.999,
    )
    print("Finished. Best:", out["best_path"], "Last:", out["last_path"])


if __name__ == "__main__":
    main()
