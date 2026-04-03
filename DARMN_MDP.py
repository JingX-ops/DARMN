import os
import re
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_txt_matrix(path: str, dtype=np.float32) -> np.ndarray:
    # txt is space-separated 0/1 matrix
    return np.loadtxt(path, dtype=dtype)

def find_feature_files(folder: str, uniprot: str) -> Tuple[str, str]:
    """
    In ./feature_shiyan/<uniprot>/ there are 2 npy:
      *single_repr*.npy
      *pair_repr*.npy
    Return (single_path, pair_path).
    """
    sub = os.path.join(folder, uniprot)
    if not os.path.isdir(sub):
        raise FileNotFoundError(f"Feature folder missing: {sub}")

    files = [f for f in os.listdir(sub) if f.endswith(".npy")]
    single = [f for f in files if "single_repr" in f]
    pair = [f for f in files if "pair_repr" in f]
    if len(single) != 1 or len(pair) != 1:
        raise RuntimeError(
            f"Expect exactly 1 single & 1 pair npy in {sub}, got single={single}, pair={pair}"
        )
    return os.path.join(sub, single[0]), os.path.join(sub, pair[0])

def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return a / (b + eps)

def sigmoid_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def flatten_valid(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
    """
    pred/target/valid_mask: (B, Lmax, Lmax)
    Return 1D tensors filtered by valid_mask==1
    """
    m = valid_mask > 0.5
    p = pred[m]
    y = target[m]
    return p.detach().cpu().numpy(), y.detach().cpu().numpy()

def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUPRC (average precision area) without sklearn.
    - y_true: {0,1}
    - y_score: [0,1]
    """
    if y_true.size == 0:
        return float("nan")
    # sort by score desc
    order = np.argsort(-y_score)
    y = y_true[order]
    # cumulative TP/FP
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    denom = (tp + fp)
    precision = tp / np.maximum(denom, 1)
    recall = tp / max(int((y_true == 1).sum()), 1)

    # AP-style area: sum over recall increments weighted by precision at that point
    # This equals average precision for binary labels
    # delta recall occurs when y==1
    ap = 0.0
    prev_recall = 0.0
    for i in range(len(y)):
        if y[i] == 1:
            r = recall[i]
            ap += precision[i] * (r - prev_recall)
            prev_recall = r
    return float(ap)

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """
    Precision/Recall/F1/MCC at threshold.
    """
    if y_true.size == 0:
        return {k: float("nan") for k in ["precision","recall","f1","mcc"]}

    y_pred = (y_prob >= thr).astype(np.int32)
    y_true = y_true.astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0

    # MCC
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = ((tp*tn - fp*fn) / denom) if denom > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "mcc": mcc}

def collect_predictions(model, loader, device):
    model.eval()
    all_prob = []
    all_true = []

    for single, pair, y, intra, inter, valid, uniprots, lengths in loader:
        single = single.to(device)
        pair = pair.to(device)
        y = y.to(device)
        valid = valid.to(device)

        pred = model(single, pair)  # (B,L,L)

        p_flat, y_flat = flatten_valid(pred, y, valid)
        all_prob.append(p_flat)
        all_true.append(y_flat)

    if len(all_prob) == 0:
        return np.array([]), np.array([])

    y_prob = np.concatenate(all_prob, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    return y_true, y_prob

def sweep_threshold_by_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float = 0.01
):
    """
    Sweep threshold on validation set.
    Return:
      best_thr
      best_metrics (dict)
      curve: list of (thr, recall, precision)
    """
    thresholds = np.arange(0.0, 1.0 + 1e-8, step)

    best_f1 = -1.0
    best_thr = 0.5
    best_metrics = None

    curve = []

    for thr in thresholds:
        m = compute_binary_metrics(y_true, y_prob, thr=thr)

        curve.append((thr, m["recall"], m["precision"]))

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr
            best_metrics = m

    return best_thr, best_metrics, curve

def save_pr_curve_txt(curve, path):
    """
    curve: list of (thr, recall, precision)
    """
    with open(path, "w") as f:
        f.write("# threshold recall precision\n")
        for thr, recall, precision in curve:
            f.write(f"{thr:.2f} {recall:.6f} {precision:.6f}\n")

def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUROC without sklearn.
    - y_true: {0,1}
    - y_score: [0,1]
    """
    if y_true.size == 0:
        return float("nan")

    # sort by score ascending
    order = np.argsort(y_score)
    y = y_true[order]

    pos = (y == 1)
    neg = (y == 0)

    n_pos = pos.sum()
    n_neg = neg.sum()

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # rank-based AUROC (equivalent to Mann–Whitney U)
    rank = np.arange(1, len(y) + 1)
    sum_ranks_pos = rank[pos].sum()

    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

#import pred results
@torch.no_grad()
def export_test_predictions_txt(model, loader, device, out_dir="./pred"):

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    for single, pair, y, intra, inter, valid, uniprots, lengths in loader:
        single = single.to(device)
        pair = pair.to(device)

        pred = model(single, pair)  # (B, Lmax, Lmax), in [0,1]

        B = pred.shape[0]
        for b in range(B):
            uid = uniprots[b]
            L = int(lengths[b])

            # 裁剪到真实尺寸 LxL（去掉 padding）
            p = pred[b, :L, :L].detach().cpu().numpy()

            # 写入 txt：空格分隔
            out_path = os.path.join(out_dir, f"{uid}.txt")
            np.savetxt(out_path, p, fmt="%.6f", delimiter=" ")


@torch.no_grad()
def collect_predictions_long_range_only_test(model, loader, device, min_seq_sep=20):
    """
    只用于 TEST 阶段：不使用 valid_mask
    通过 lengths 把 padding 裁掉，然后用 distance mask (|i-j|>min_seq_sep) 筛选。
    返回 1D: y_true_lr, y_prob_lr
    """
    model.eval()
    all_true = []
    all_prob = []

    for single, pair, y, intra, inter, valid, uniprots, lengths in loader:
        single = single.to(device)
        pair = pair.to(device)
        y = y.to(device)

        pred = model(single, pair)  # (B, Lmax, Lmax)

        B = pred.shape[0]
        for b in range(B):
            L = int(lengths[b])

            # 裁剪到真实 L×L（不使用 valid_mask）
            y_b = y[b, :L, :L]
            p_b = pred[b, :L, :L]

            # distance mask: |i-j| > min_seq_sep
            idx = torch.arange(L, device=device)
            dist = torch.abs(idx[None, :] - idx[:, None])
            dist_mask = dist > min_seq_sep  # (L,L) bool

            y_flat = y_b[dist_mask].detach().cpu().numpy()
            p_flat = p_b[dist_mask].detach().cpu().numpy()

            if y_flat.size > 0:
                all_true.append(y_flat)
                all_prob.append(p_flat)

    if len(all_true) == 0:
        return np.array([]), np.array([])

    y_true_lr = np.concatenate(all_true, axis=0)
    y_prob_lr = np.concatenate(all_prob, axis=0)
    return y_true_lr, y_prob_lr


# =========================
# Dataset & Collate (pad to batch max L)
# =========================
@dataclass
class ProteinSample:
    uniprot: str
    single: np.ndarray   # (L,256)
    pair: np.ndarray     # (L,L,128)
    y: np.ndarray        # (L,L) 0/1
    mask_intra: np.ndarray  # (L,L) 0/1
    mask_inter: np.ndarray  # (L,L) 0/1

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str):
        """
        split: 'train' | 'val' | 'test'
        Expect directories:
          dcm_moni_{split}
          feature_moni_{split}
          mask_moni_{split}
        """
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split

        self.dcm_dir = os.path.join(root, f"dcm_moni_{split}")
        self.feat_dir = os.path.join(root, f"feature_moni_{split}")
        self.mask_dir = os.path.join(root, f"mask_moni_{split}")

        if not os.path.isdir(self.dcm_dir):
            raise FileNotFoundError(self.dcm_dir)
        if not os.path.isdir(self.feat_dir):
            raise FileNotFoundError(self.feat_dir)
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(self.mask_dir)

        # uniprot ids from dcm_moni_{split}/*.txt
        self.ids = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.dcm_dir)
            if f.endswith(".txt")
        )

        if len(self.ids) == 0:
            raise RuntimeError(f"No label txt found in {self.dcm_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx) -> ProteinSample:
        uniprot = self.ids[idx]

        # label
        y_path = os.path.join(self.dcm_dir, f"{uniprot}.txt")
        y = load_txt_matrix(y_path, dtype=np.float32)

        # masks
        inter_path = os.path.join(self.mask_dir, f"{uniprot}_inter.txt")
        intra_path = os.path.join(self.mask_dir, f"{uniprot}_intra.txt")
        if not os.path.exists(inter_path) or not os.path.exists(intra_path):
            raise FileNotFoundError(
                f"mask missing for {uniprot}: {inter_path} / {intra_path}"
            )

        mask_inter = load_txt_matrix(inter_path, dtype=np.float32)
        mask_intra = load_txt_matrix(intra_path, dtype=np.float32)

        # features
        single_path, pair_path = find_feature_files(self.feat_dir, uniprot)
        single = np.load(single_path).astype(np.float32)
        pair = np.load(pair_path).astype(np.float32)

        # sanity checks
        L = single.shape[0]
        assert single.shape == (L, 256)
        assert pair.shape == (L, L, 128)
        assert y.shape == (L, L)
        assert mask_intra.shape == (L, L)
        assert mask_inter.shape == (L, L)

        return ProteinSample(uniprot, single, pair, y, mask_intra, mask_inter)


def collate_pad(batch: List[ProteinSample]):
    """
    Pad all proteins in batch to max L in this batch.
    Return tensors:
      single: (B, Lmax, 256)
      pair:   (B, Lmax, Lmax, 128)
      y:      (B, Lmax, Lmax)
      intra/inter: (B, Lmax, Lmax)
      valid_mask:  (B, Lmax, Lmax) 1 if inside original L, else 0
      uniprots: list
      lengths: list
    """
    B = len(batch)
    lengths = [s.single.shape[0] for s in batch]
    Lmax = max(lengths)

    single = np.zeros((B, Lmax, 256), dtype=np.float32)
    pair   = np.zeros((B, Lmax, Lmax, 128), dtype=np.float32)
    y      = np.zeros((B, Lmax, Lmax), dtype=np.float32)
    intra  = np.zeros((B, Lmax, Lmax), dtype=np.float32)
    inter  = np.zeros((B, Lmax, Lmax), dtype=np.float32)
    valid  = np.zeros((B, Lmax, Lmax), dtype=np.float32)

    uniprots = []
    for b, s in enumerate(batch):
        L = s.single.shape[0]
        uniprots.append(s.uniprot)

        single[b, :L, :] = s.single
        pair[b, :L, :L, :] = s.pair
        y[b, :L, :L] = s.y
        intra[b, :L, :L] = s.mask_intra
        inter[b, :L, :L] = s.mask_inter
        valid[b, :L, :L] = 1.0

    # torch
    return (
        torch.from_numpy(single),
        torch.from_numpy(pair),
        torch.from_numpy(y),
        torch.from_numpy(intra),
        torch.from_numpy(inter),
        torch.from_numpy(valid),
        uniprots,
        lengths
    )


# =========================
# Model: Residual FiLM (trainable) + Row/Col MHSA + MLP head
# =========================
class ResidualFiLM(nn.Module):
    """
    Trainable Residual FiLM:
      pair' = gamma * pair + beta
      gamma = 1 + alpha * tanh(gamma_hat)
      beta  = alpha * tanh(beta_hat)

    Condition uses [S_i, S_j] -> (B,L,L,512)
    """
    def __init__(self, single_dim=256, pair_dim=128, hidden=128, alpha=0.2):
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.hidden = hidden
        self.alpha = alpha

        self.fc1 = nn.Linear(2 * single_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 * pair_dim)

    def forward(self, single: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        """
        single: (B,L,256)
        pair:   (B,L,L,128)
        """
        B, L, _ = single.shape
        assert pair.shape[:3] == (B, L, L)

        # build C_ij = [S_i, S_j]
        Si = single[:, :, None, :].expand(B, L, L, self.single_dim)
        Sj = single[:, None, :, :].expand(B, L, L, self.single_dim)
        C = torch.cat([Si, Sj], dim=-1)  # (B,L,L,512)

        x = F.relu(self.fc1(C))
        x = self.fc2(x)  # (B,L,L,256)

        gamma_hat = x[..., :self.pair_dim]
        beta_hat  = x[..., self.pair_dim:]

        gamma = 1.0 + self.alpha * torch.tanh(gamma_hat)
        beta  = self.alpha * torch.tanh(beta_hat)

        return gamma * pair + beta

class RowColMultiHeadAttention(nn.Module):
    """
      Input:  (B,L,L,C)
      Row attention: treat rows as batch -> (B*L, L, C)
      Col attention: transpose then same
    """
    def __init__(self, dim=128, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def _mhsa(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,H,N,d)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,N,d)
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B,H,N,d)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        # pair: (B,L,L,C)
        B, L, _, C = pair.shape
        assert C == self.dim

        # Row-wise: (B*L, L, C)
        row_in = pair.reshape(B * L, L, C)
        row_out = self._mhsa(row_in).reshape(B, L, L, C)

        # Col-wise
        col_in = row_out.transpose(1, 2).reshape(B * L, L, C)
        col_out = self._mhsa(col_in).reshape(B, L, L, C).transpose(1, 2)

        return col_out

#add
class BlockRowColMultiHeadAttention(nn.Module):
    def __init__(self, dim=32, num_heads=2, window=64):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window = window

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def _mhsa(self, x):
        # x: (Bblk, W, C)
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = torch.softmax(att, dim=-1)
        out = att @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)

    def forward(self, pair):
        """
        pair: (B, L, L, C)
        """
        B, L, _, C = pair.shape
        w = self.window

        # ========== Row-wise ==========
        # (B, L, L, C) -> (B*L, L, C)
        row = pair.reshape(B * L, L, C)

        # pad to multiple of window
        pad = (w - L % w) % w
        if pad > 0:
            row = F.pad(row, (0, 0, 0, pad))  # pad N dim

        Lp = row.shape[1]
        nblk = Lp // w

        # (B*L, nblk, w, C) -> (B*L*nblk, w, C)
        row_blk = row.view(B * L, nblk, w, C).reshape(-1, w, C)

        row_blk = self._mhsa(row_blk)

        # reshape back
        row = row_blk.view(B * L, nblk, w, C).reshape(B * L, Lp, C)
        row = row[:, :L, :]   # remove padding
        row = row.view(B, L, L, C)

        # ========== Col-wise ==========
        col = row.transpose(1, 2).reshape(B * L, L, C)

        if pad > 0:
            col = F.pad(col, (0, 0, 0, pad))

        col_blk = col.view(B * L, nblk, w, C).reshape(-1, w, C)
        col_blk = self._mhsa(col_blk)

        col = col_blk.view(B * L, nblk, w, C).reshape(B * L, Lp, C)
        col = col[:, :L, :]
        col = col.view(B, L, L, C).transpose(1, 2)

        return col

class PairMLPHead(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, pair_feat: torch.Tensor) -> torch.Tensor:
        # (B,L,L,C) -> (B,L,L)
        logits = self.mlp(pair_feat).squeeze(-1)
        return sigmoid_safe(logits)

class DynamicContactNet(nn.Module):
    def __init__(self, film_hidden=128, film_alpha=0.2, attn_heads=8, mlp_hidden=64):
        super().__init__()
        self.film = ResidualFiLM(hidden=film_hidden, alpha=film_alpha)
        self.attn = BlockRowColMultiHeadAttention(dim=32, num_heads=2, window=64)
        self.head = PairMLPHead(in_dim=32, hidden_dim=mlp_hidden)
        self.pair_reduce = nn.Sequential(nn.Linear(128, 64),nn.GELU(),nn.Linear(64, 32))

    def forward(self, single: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        # single: (B,L,256), pair: (B,L,L,128)
        pair2 = self.film(single, pair)
        pair2 = self.pair_reduce(pair2)
        pair3 = self.attn(pair2)
        pred = self.head(pair3)  # (B,L,L) in [0,1]
        return pred


# =========================
# Loss: Domain-aware weighted focal + valid_mask
# =========================
def focal_loss_elementwise(pred, target, alpha=0.75, gamma=2.0, eps=1e-8):
    # pred/target: (B,L,L)
    pred = pred.clamp(min=eps, max=1.0 - eps)
    p_t = torch.where(target > 0.5, pred, 1.0 - pred)

    # alpha_t as scalar (broadcast)
    alpha_t = torch.where(target > 0.5,
                          torch.full_like(pred, fill_value=alpha),
                          torch.full_like(pred, fill_value=1.0 - alpha))
    loss = -alpha_t * (1.0 - p_t) ** gamma * torch.log(p_t)
    return loss

class DomainAwareWeightedFocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.85, gamma=2.0,
                 lambda_intra=1.0, lambda_inter=3.0,
                 eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.eps = eps

    def forward(self,
                pred, target,
                mask_intra, mask_inter,
                valid_mask):
        """
        pred/target/masks/valid_mask: (B,L,L)
        valid_mask: masks out padded region
        """
        fl = focal_loss_elementwise(pred, target, self.alpha, self.gamma, self.eps)

        # apply valid_mask to domain masks
        mi = mask_intra * valid_mask
        me = mask_inter * valid_mask

        intra = safe_div((fl * mi).sum(), mi.sum(), self.eps)
        inter = safe_div((fl * me).sum(), me.sum(), self.eps)

        loss = self.lambda_intra * intra + self.lambda_inter * inter
        return loss, {"intra_loss": intra.detach(), "inter_loss": inter.detach()}


# =========================
# Train / Eval
# =========================
@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    all_prob = []
    all_true = []

    for single, pair, y, intra, inter, valid, uniprots, lengths in loader:
        single = single.to(device)
        pair = pair.to(device)
        y = y.to(device)
        intra = intra.to(device)
        inter = inter.to(device)
        valid = valid.to(device)

        pred = model(single, pair)  # (B,L,L)

        p_flat, y_flat = flatten_valid(pred, y, valid)
        all_prob.append(p_flat)
        all_true.append(y_flat)

    y_prob = np.concatenate(all_prob, axis=0) if len(all_prob) else np.array([])
    y_true = np.concatenate(all_true, axis=0) if len(all_true) else np.array([])

    auprc = compute_pr_auc(y_true, y_prob)
    m = compute_binary_metrics(y_true, y_prob, thr=thr)
    m["auprc"] = auprc
    return m

def train_one_epoch(model, criterion, optimizer, loader, device):
    model.train()
    total = 0.0
    n = 0

    for single, pair, y, intra, inter, valid, uniprots, lengths in loader:
        single = single.to(device)
        pair = pair.to(device)
        y = y.to(device)
        intra = intra.to(device)
        inter = inter.to(device)
        valid = valid.to(device)

        pred = model(single, pair)
        loss, aux = criterion(pred, y, intra, inter, valid)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(n, 1)

@torch.no_grad()
def compute_val_loss(model, criterion, loader, device):
    model.eval()
    total = 0.0
    n = 0

    for single, pair, y, intra, inter, valid, _, _ in loader:
        single = single.to(device)
        pair = pair.to(device)
        y = y.to(device)
        intra = intra.to(device)
        inter = inter.to(device)
        valid = valid.to(device)

        pred = model(single, pair)
        loss, _ = criterion(pred, y, intra, inter, valid)

        total += float(loss.item())
        n += 1

    return total / max(n, 1)

def split_ids(ids: List[str], seed=42, train_ratio=0.8, val_ratio=0.1):
    ids = ids[:]
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train+n_val]
    test_ids = ids[n_train+n_val:]
    return train_ids, val_ids, test_ids

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, base: ProteinDataset, keep_ids: List[str]):
        self.base = base
        self.id_to_idx = {uid: i for i, uid in enumerate(base.ids)}
        self.indices = [self.id_to_idx[uid] for uid in keep_ids]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./", help="project root directory")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=5, help="A100，5")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thr", type=float, default=0.5, help="threshold for P/R/F1/MCC")
    ap.add_argument("--save_dir", type=str, default="./ckpt_moni")
    ap.add_argument("--resume", type=str, default="", help="path to .pt to resume")
    # model hparams
    ap.add_argument("--film_alpha", type=float, default=0.2)
    ap.add_argument("--film_hidden", type=int, default=128)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--mlp_hidden", type=int, default=64)
    # loss hparams
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--lambda_intra", type=float, default=1.0)
    ap.add_argument("--lambda_inter", type=float, default=3.0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    train_ds = ProteinDataset(args.root, split="train")
    val_ds   = ProteinDataset(args.root, split="val")
    test_ds  = ProteinDataset(args.root, split="test")

    print(
        f"[Data] "
        f"train={len(train_ds)} "
        f"val={len(val_ds)} "
        f"test={len(test_ds)}"
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pad, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pad, num_workers=0
    ) if val_ds is not None else None
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pad, num_workers=0
    ) if test_ds is not None else None

    model = DynamicContactNet(
        film_hidden=args.film_hidden,
        film_alpha=args.film_alpha,
        attn_heads=args.attn_heads,
        mlp_hidden=args.mlp_hidden
    ).to(device)

    PRETRAINED_CKPT = "./ckpt_shiyan/best.pt"

    if os.path.exists(PRETRAINED_CKPT):
        ckpt = torch.load(PRETRAINED_CKPT, map_location=device)

        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt  # safety fallback

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"[Pretrained] Loaded model weights from {PRETRAINED_CKPT}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected}")
    else:
        print(f"[Pretrained] WARNING: {PRETRAINED_CKPT} not found, training from scratch")

    criterion = DomainAwareWeightedFocalLoss(
        alpha=args.alpha,
        gamma=args.gamma,
        lambda_intra=args.lambda_intra,
        lambda_inter=args.lambda_inter
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_val_auprc = -1.0

    # resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_auprc = ckpt.get("best_val_auprc", -1.0)
        print(f"[Resume] from {args.resume}, start_epoch={start_epoch}, best_val_auprc={best_val_auprc}")

    # train loop
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)

        msg = f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f}"

        if val_loader is not None:
            val_loss = compute_val_loss(model, criterion, val_loader, device)
            msg += f" | val_loss={val_loss:.6f}"
            # ===== Validation: threshold selection by best F1 =====
            y_true, y_prob = collect_predictions(model, val_loader, device)

            # threshold-free metric
            val_auprc = compute_pr_auc(y_true, y_prob)
            val_auroc = compute_roc_auc(y_true, y_prob)

            # sweep threshold
            best_thr, best_m, curve = sweep_threshold_by_f1(
                y_true, y_prob, step=0.01
            )

            msg += (
                f" | val_AUPRC={val_auprc:.4f}"
                f" | val_AUROC={val_auroc:.4f}"
                f" | best_thr={best_thr:.2f}"
                f" P={best_m['precision']:.4f}"
                f" R={best_m['recall']:.4f}"
                f" F1={best_m['f1']:.4f}"
                f" MCC={best_m['mcc']:.4f}"
            )
            
            # save best
            if not math.isnan(val_auprc) and val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                save_path = os.path.join(args.save_dir, "best.pt")

                torch.save({
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_auprc": best_val_auprc,
                    "best_thr": best_thr,   # ⭐ 保存阈值
                    "args": vars(args),
                }, save_path)

                msg += f"  [SAVE best -> {save_path}]"

        else:
            # no val set, still save each epoch
            save_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}.pt")
            torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, save_path)
            msg += f"  [SAVE -> {save_path}]"

        print(msg)

    # final test with best
    best_path = os.path.join(args.save_dir, "best.pt")

    if test_loader is not None and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        export_test_predictions_txt(model, test_loader, device, out_dir="./pred")
        print("[Export] Saved test predictions to ./pred/*.txt")

        # 来自 validation 的冻结 threshold
        best_thr = ckpt.get("best_thr", 0.5)

        # test set predictions
        y_true, y_prob = collect_predictions(model, test_loader, device)

        # threshold-free
        test_auprc = compute_pr_auc(y_true, y_prob)
        test_auroc = compute_roc_auc(y_true, y_prob)

        # thresholded metrics
        test_m = compute_binary_metrics(y_true, y_prob, thr=best_thr)

        print(
            f"[TEST best] "
            f"AUPRC={test_auprc:.4f} "
            f"AUROC={test_auroc:.4f} "
            f"thr={best_thr:.2f} "
            f"P={test_m['precision']:.4f} "
            f"R={test_m['recall']:.4f} "
            f"F1={test_m['f1']:.4f} "
            f"MCC={test_m['mcc']:.4f}"
        )

        # ===== Long-range eval on TEST: |i-j| > 20 (distance mask only) =====
        y_true_lr, y_prob_lr = collect_predictions_long_range_only_test(
            model, test_loader, device, min_seq_sep=20
        )

        test_lr_auprc = compute_pr_auc(y_true_lr, y_prob_lr)
        test_lr_auroc = compute_roc_auc(y_true_lr, y_prob_lr)
        test_lr_m = compute_binary_metrics(y_true_lr, y_prob_lr, thr=best_thr)

        print(
            f"[TEST long-range |i-j|>20] "
            f"AUPRC={test_lr_auprc:.4f} "
            f"AUROC={test_lr_auroc:.4f} "
            f"thr={best_thr:.2f} "
            f"P={test_lr_m['precision']:.4f} "
            f"R={test_lr_m['recall']:.4f} "
            f"F1={test_lr_m['f1']:.4f} "
            f"MCC={test_lr_m['mcc']:.4f}"
        )

        # PR curve txt for long-range (same format as your full-distance one)
        _, _, test_lr_curve = sweep_threshold_by_f1(y_true_lr, y_prob_lr, step=0.01)
        lr_pr_curve_path = os.path.join(args.save_dir, "test_pr_curve_long_range_gt20.txt")
        save_pr_curve_txt(test_lr_curve, lr_pr_curve_path)


        # ===== PR curve on TEST set (for plotting only) =====
        _, _, test_curve = sweep_threshold_by_f1(
            y_true, y_prob, step=0.01
        )

        pr_curve_path = os.path.join(
            args.save_dir,
            "test_pr_curve.txt"
        )
        save_pr_curve_txt(test_curve, pr_curve_path)

if __name__ == "__main__":
    main()
