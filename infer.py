#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 默认参数
# =========================

UNIPROT_ID = "Q2I742"

CKPT_PATH = "./ckpt_moni/best.pt"

FEATURE_DIR = "./feature_shiyan_train"
MASK_DIR = "./mask_shiyan_train"
DCM_DIR = "./dcm_shiyan_train"

OUT_DIR = "./pred_Q2I742"
OUT_FILE = "Q2I742.txt"

SEED = 42

# 与训练脚本保持一致
FILM_ALPHA = 0.2
FILM_HIDDEN = 128
MLP_HIDDEN = 64


# =========================
# Utils
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_txt_matrix(path: str, dtype=np.float32) -> np.ndarray:
    return np.loadtxt(path, dtype=dtype)


def find_feature_files(folder: str, uniprot: str) -> Tuple[str, str]:
    sub = os.path.join(folder, uniprot)

    if not os.path.isdir(sub):
        raise FileNotFoundError(f"Feature folder missing: {sub}")

    files = [f for f in os.listdir(sub) if f.endswith(".npy")]
    single = [f for f in files if "single_repr" in f]
    pair = [f for f in files if "pair_repr" in f]

    if len(single) != 1 or len(pair) != 1:
        raise RuntimeError(
            f"Expect exactly 1 single_repr and 1 pair_repr npy in {sub}, "
            f"got single={single}, pair={pair}"
        )

    return os.path.join(sub, single[0]), os.path.join(sub, pair[0])


def sigmoid_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


# =========================
# Model
# 与训练脚本保持一致
# =========================

class ResidualFiLM(nn.Module):
    def __init__(self, single_dim=256, pair_dim=128, hidden=128, alpha=0.2):
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.hidden = hidden
        self.alpha = alpha

        self.fc1 = nn.Linear(2 * single_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 * pair_dim)

    def forward(self, single: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        B, L, _ = single.shape

        Si = single[:, :, None, :].expand(B, L, L, self.single_dim)
        Sj = single[:, None, :, :].expand(B, L, L, self.single_dim)
        C = torch.cat([Si, Sj], dim=-1)

        x = F.relu(self.fc1(C))
        x = self.fc2(x)

        gamma_hat = x[..., :self.pair_dim]
        beta_hat = x[..., self.pair_dim:]

        gamma = 1.0 + self.alpha * torch.tanh(gamma_hat)
        beta = self.alpha * torch.tanh(beta_hat)

        return gamma * pair + beta


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
        B, L, _, C = pair.shape
        w = self.window

        row = pair.reshape(B * L, L, C)

        pad = (w - L % w) % w
        if pad > 0:
            row = F.pad(row, (0, 0, 0, pad))

        Lp = row.shape[1]
        nblk = Lp // w

        row_blk = row.view(B * L, nblk, w, C).reshape(-1, w, C)
        row_blk = self._mhsa(row_blk)

        row = row_blk.view(B * L, nblk, w, C).reshape(B * L, Lp, C)
        row = row[:, :L, :]
        row = row.view(B, L, L, C)

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
    def __init__(self, in_dim=32, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_feat: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(pair_feat).squeeze(-1)
        return sigmoid_safe(logits)


class DynamicContactNet(nn.Module):
    def __init__(self, film_hidden=128, film_alpha=0.2, attn_heads=8, mlp_hidden=64):
        super().__init__()

        # attn_heads 参数保留是为了与训练脚本接口一致；
        # 实际 block attention 使用 num_heads=2，与训练脚本一致。
        self.film = ResidualFiLM(hidden=film_hidden, alpha=film_alpha)

        self.pair_reduce = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        self.attn = BlockRowColMultiHeadAttention(dim=32, num_heads=2, window=64)
        self.head = PairMLPHead(in_dim=32, hidden_dim=mlp_hidden)

    def forward(self, single: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        pair2 = self.film(single, pair)
        pair2 = self.pair_reduce(pair2)
        pair3 = self.attn(pair2)
        pred = self.head(pair3)
        return pred


# =========================
# Data loading
# =========================

def load_q2i742_inputs():
    uid = UNIPROT_ID

    y_path = os.path.join(DCM_DIR, f"{uid}.txt")
    intra_path = os.path.join(MASK_DIR, f"{uid}_intra.txt")
    inter_path = os.path.join(MASK_DIR, f"{uid}_inter.txt")

    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Label missing: {y_path}")
    if not os.path.exists(intra_path):
        raise FileNotFoundError(f"Intra mask missing: {intra_path}")
    if not os.path.exists(inter_path):
        raise FileNotFoundError(f"Inter mask missing: {inter_path}")

    y = load_txt_matrix(y_path, dtype=np.float32)
    mask_intra = load_txt_matrix(intra_path, dtype=np.float32)
    mask_inter = load_txt_matrix(inter_path, dtype=np.float32)

    single_path, pair_path = find_feature_files(FEATURE_DIR, uid)

    single = np.load(single_path).astype(np.float32)
    pair = np.load(pair_path).astype(np.float32)

    L = single.shape[0]

    if single.shape != (L, 256):
        raise ValueError(f"{uid}: single shape invalid: {single.shape}; expected {(L, 256)}")
    if pair.shape != (L, L, 128):
        raise ValueError(f"{uid}: pair shape invalid: {pair.shape}; expected {(L, L, 128)}")
    if y.shape != (L, L):
        raise ValueError(f"{uid}: label shape invalid: {y.shape}; expected {(L, L)}")
    if mask_intra.shape != (L, L):
        raise ValueError(f"{uid}: intra mask shape invalid: {mask_intra.shape}; expected {(L, L)}")
    if mask_inter.shape != (L, L):
        raise ValueError(f"{uid}: inter mask shape invalid: {mask_inter.shape}; expected {(L, L)}")

    print(f"[Input] {uid}")
    print(f"  single: {single.shape} <- {single_path}")
    print(f"  pair  : {pair.shape} <- {pair_path}")
    print(f"  label : {y.shape} <- {y_path}")
    print(f"  intra : {mask_intra.shape} <- {intra_path}")
    print(f"  inter : {mask_inter.shape} <- {inter_path}")

    return single, pair, L


def load_model(device: str):
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model = DynamicContactNet(
        film_hidden=FILM_HIDDEN,
        film_alpha=FILM_ALPHA,
        mlp_hidden=MLP_HIDDEN,
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # 兼容 DataParallel 保存的 module.xxx
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch", "NA")
        best_val_auprc = ckpt.get("best_val_auprc", "NA")
        best_thr = ckpt.get("best_thr", "NA")
        print(f"[Checkpoint] loaded: {CKPT_PATH}")
        print(f"  epoch={epoch}, best_val_auprc={best_val_auprc}, best_thr={best_thr}")
    else:
        print(f"[Checkpoint] loaded state_dict: {CKPT_PATH}")

    return model


@torch.no_grad()
def predict_one_system(model, single_np: np.ndarray, pair_np: np.ndarray, device: str) -> np.ndarray:
    single = torch.from_numpy(single_np).unsqueeze(0).to(device)
    pair = torch.from_numpy(pair_np).unsqueeze(0).to(device)

    pred = model(single, pair)[0]
    prob = pred.detach().cpu().numpy()

    return prob


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    os.makedirs(OUT_DIR, exist_ok=True)

    single, pair, L = load_q2i742_inputs()
    model = load_model(device)

    prob = predict_one_system(model, single, pair, device)

    if prob.shape != (L, L):
        raise RuntimeError(f"Prediction shape invalid: {prob.shape}; expected {(L, L)}")

    out_path = os.path.join(OUT_DIR, OUT_FILE)
    np.savetxt(out_path, prob, fmt="%.6f", delimiter=" ")

    print("[Done]")
    print(f"Prediction probability matrix saved to: {out_path}")
    print(f"Shape: {prob.shape}")
    print("This is a probability matrix, not a binary matrix.")


if __name__ == "__main__":
    main()
