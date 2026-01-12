import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.07,
                 class_balance: bool = False,
                 eps: float = 1e-12,
                 assert_unit_norm: bool = False,
                 norm_tol: float = 1e-3,
                 weak_same_alpha: float = 1e-5,
                 ):
        super().__init__()
        self.tau = float(temperature)
        self.class_balance = class_balance
        self.eps = eps
        self.assert_unit_norm = assert_unit_norm
        self.norm_tol = norm_tol
        self.alpha = float(weak_same_alpha)
        self._last_stats = {}

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        assert z1.dim() == 2 and z2.dim() == 2, f"z1/z2 must be [B,d], got {z1.shape}, {z2.shape}"
        assert z1.shape == z2.shape, "z1 and z2 must have the same shape"
        B, d = z1.shape
        device = z1.device


        if self.assert_unit_norm:
            with torch.no_grad():
                n1 = z1.norm(dim=-1)
                n2 = z2.norm(dim=-1)
                if not (torch.allclose(n1, torch.ones_like(n1), atol=self.norm_tol) and
                        torch.allclose(n2, torch.ones_like(n2), atol=self.norm_tol)):
                    raise ValueError("z1/z2 are expected to be L2-normalized already.")


        Z = torch.cat([z1, z2], dim=0)
        y = torch.cat([labels, labels], dim=0)


        Zf = Z.float()
        sim = (Zf @ Zf.t()) / self.tau
        self_mask = torch.eye(2 * B, device=device, dtype=torch.bool)
        logits = sim.masked_fill(self_mask, float('-inf'))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = log_prob.masked_fill(self_mask, 0.0)

        idx = torch.arange(B, device=device)

        pair_mask = torch.zeros((2 * B, 2 * B), device=device, dtype=torch.bool)
        pair_mask[idx, idx + B] = True
        pair_mask[idx + B, idx] = True


        same = y[:, None].eq(y[None, :]) & (~self_mask)

        pos_weight = pair_mask.float() + self.alpha * ((same & (~pair_mask)).float())

        sum_log_pos = (log_prob * pos_weight).sum(dim=1)  # [2B]
        pos_count = pos_weight.sum(dim=1).clamp_min(1.0)  # [2B]
        mean_log_pos = sum_log_pos / pos_count
        loss = (-mean_log_pos).mean()


        with torch.no_grad():
            pos_cos = (Zf @ Zf.t()).masked_fill(~same, 0).sum() / same.sum().clamp_min(1)
            neg_mask = (~same) & (~self_mask)
            neg_cos = (Zf @ Zf.t()).masked_fill(~neg_mask, 0).sum() / neg_mask.sum().clamp_min(1)
            self._last_stats = {
                "temperature": float(self.tau),
                "alpha": float(self.alpha),
                "pos_count_mean": pos_count.mean().item(),
                "pos_cos_mean": pos_cos.item(),
                "neg_cos_mean": neg_cos.item(),
            }
        return loss
