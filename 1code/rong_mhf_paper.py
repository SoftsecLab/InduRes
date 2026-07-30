#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-aligned InduRes MHF-style three-path fusion.

Three paths:
1. Structural Evolution Path
2. Perturbation-Induced Revision Path
   = revision stability features + perturbation correction features
3. Probabilistic Response Path

Fusion:
path-specific encoders
+ cross-path attention
+ adaptive path gating
+ final classifier

This replaces feature-level concatenation with paper-style multi-path fusion.
"""

import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ============================================================
# 1. Default paths
# ============================================================

DEFAULT_BASE_TRAIN_CSV = (
    "/home/chy/1/data/hc3_processed/analysis/"
    "mhf_struct_prob/train700_struct_quick_prob_merged.csv"
)

DEFAULT_BASE_TEST_CSV = (
    "/home/chy/1/data/hc3_processed/analysis/"
    "mhf_struct_prob/test200_struct_quick_prob_merged.csv"
)

DEFAULT_PERTURB_TRAIN_CSV = (
    "/home/chy/1/data/hc3_processed/"
    "correction_metrics_v2/"
    "train700_source_features_v2.csv"
)

DEFAULT_PERTURB_TEST_CSV = (
    "/home/chy/1/data/hc3_processed/"
    "correction_metrics_v2/"
    "test200_source_features_v2.csv"
)

DEFAULT_OUT_DIR = (
    "/home/chy/1/data/hc3_processed/analysis/"
    "mhf_paper_three_path_fusion"
)


# ============================================================
# 2. Feature definitions
# ============================================================

STRUCTURAL_FEATURES = [
    "SRM_mean",
    "SRM_std",
    "SRM_max",
    "EDC_mean",
    "EDC_std",
    "EDC_max",
    "PDJ_mean",
    "PDJ_std",
    "PDJ_max",
    "PathVar",
]

QUICK_FEATURES = [
    "rewrite_len_mean",
    "rewrite_len_std",
    "source_rewrite_jaccard_mean",
    "source_rewrite_jaccard_std",
    "rewrite_pair_jaccard_mean",
    "rewrite_pair_jaccard_std",
]

PROB_FEATURES = [
    "Dfreq_src_to_human_ref",
    "Dfreq_src_to_machine_ref",
    "Dfreq_ref_margin_machine_positive",
    "Dfreq_source_rewrite_mean",
    "Dfreq_source_rewrite_std",
    "Dfreq_rewrite_ref_margin_mean",

    "Ptrans_self_rate",
    "Ptrans_changed_rate",
    "Ptrans_entropy",
    "Ptrans_entropy_norm",
    "Ptrans_max_prob",
    "Ptrans_unique_n",

    "LIV_k5",
    "LIV_k8",
    "LIV_k10",
]

PERTURBATION_FEATURES = [
    "avg_mover",
    "avg_bert",
    "structural_residual_mean",
    "corrected_semantic_fidelity_std",
    "edit_ratio_std",
    "edit_ratio_nonzero_mean",
    "semantic_drift_mean",
    "structural_drift_mean",
]

# Paper-aligned three paths
STRUCTURAL_PATH_FEATURES = STRUCTURAL_FEATURES

REVISION_PATH_FEATURES = (
    QUICK_FEATURES
    + PERTURBATION_FEATURES
)

PROB_PATH_FEATURES = PROB_FEATURES

ALL_FEATURES = (
    STRUCTURAL_PATH_FEATURES
    + REVISION_PATH_FEATURES
    + PROB_PATH_FEATURES
)


# ============================================================
# 3. Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_train_csv", default=DEFAULT_BASE_TRAIN_CSV)
    parser.add_argument("--base_test_csv", default=DEFAULT_BASE_TEST_CSV)
    parser.add_argument("--perturb_train_csv", default=DEFAULT_PERTURB_TRAIN_CSV)
    parser.add_argument("--perturb_test_csv", default=DEFAULT_PERTURB_TEST_CSV)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--heads", type=int, default=4)

    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.15)

    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


# ============================================================
# 4. Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_file(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} 不存在：\n{path}"
        )


def normalize_label_column(df, name):
    if "source_label" not in df.columns:
        raise KeyError(f"{name} 缺少 source_label")

    df["source_label"] = (
        df["source_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[df["source_label"].isin(["human", "machine"])].copy()

    if "label" not in df.columns:
        df["label"] = df["source_label"].map(
            {"human": 0, "machine": 1}
        )
    else:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df.dropna(subset=["source_label", "label"]).copy()
    df["label"] = df["label"].astype(int)

    expected = df["source_label"].map(
        {"human": 0, "machine": 1}
    ).astype(int)

    bad = df["label"] != expected
    if bad.any():
        raise ValueError(
            f"{name} 中 label 与 source_label 不一致：\n"
            f"{df.loc[bad, ['source_label', 'label']].head()}"
        )

    return df


def load_base_data(path, split_name):
    df = pd.read_csv(path)

    if "n_rewrites" in df.columns:
        df = df[df["n_rewrites"] == 3].copy()

    if "quick_n_rewrites" in df.columns:
        df = df[df["quick_n_rewrites"] == 3].copy()

    if "n_rewrites_quick" in df.columns:
        df = df[df["n_rewrites_quick"] == 3].copy()

    df = normalize_label_column(df, f"{split_name} base")
    return df


def load_perturbation_data(path, split_name):
    df = pd.read_csv(path)
    df = normalize_label_column(df, f"{split_name} perturbation")

    missing = [
        x for x in PERTURBATION_FEATURES
        if x not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{split_name} perturbation 缺少特征：\n{missing}\n"
            f"当前文件：{path}"
        )

    return df


def make_string_key(df, col):
    df[col] = df[col].astype(str).str.strip()
    return df


def assert_unique(df, keys, name):
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        raise ValueError(
            f"{name} 在键 {keys} 上不是唯一：\n"
            f"{df.loc[dup, keys].sort_values(keys).head(20)}"
        )


def determine_merge_strategy(base_df, perturb_df):
    if "topic_id" in base_df.columns and "topic_id" in perturb_df.columns:
        return {
            "mode": "topic",
            "base_keys": ["topic_id", "source_label"],
            "perturb_keys": ["topic_id", "source_label"],
        }

    if "task_id" in base_df.columns and "sample_id" in perturb_df.columns:
        return {
            "mode": "task_sample",
            "base_keys": ["task_id", "source_label"],
            "perturb_keys": ["sample_id", "source_label"],
        }

    if "sample_id" in base_df.columns and "sample_id" in perturb_df.columns:
        return {
            "mode": "sample",
            "base_keys": ["sample_id", "source_label"],
            "perturb_keys": ["sample_id", "source_label"],
        }

    raise KeyError(
        "无法确定 base 与 perturbation 的合并键。"
    )


def merge_base_and_perturbation(base_df, perturb_df, split_name):
    strategy = determine_merge_strategy(base_df, perturb_df)

    base = base_df.copy()
    perturb = perturb_df.copy()

    base_keys = strategy["base_keys"]
    perturb_keys = strategy["perturb_keys"]

    for c in base_keys:
        base = make_string_key(base, c)

    for c in perturb_keys:
        perturb = make_string_key(perturb, c)

    assert_unique(base, base_keys, f"{split_name} base")
    assert_unique(perturb, perturb_keys, f"{split_name} perturbation")

    rename_mapping = {
        p: b
        for b, p in zip(base_keys, perturb_keys)
        if p != b
    }

    perturb = perturb.rename(columns=rename_mapping)

    merge_keys = base_keys

    keep_cols = (
        merge_keys
        + ["label"]
        + PERTURBATION_FEATURES
    )

    if (
        "topic_id" not in base.columns
        and "topic_id" in perturb.columns
        and "topic_id" not in keep_cols
    ):
        keep_cols.append("topic_id")

    keep_cols = list(dict.fromkeys(keep_cols))

    perturb_selected = (
        perturb[keep_cols]
        .copy()
        .rename(columns={"label": "perturb_label"})
    )

    n0 = len(base)

    merged = base.merge(
        perturb_selected,
        on=merge_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if len(merged) != n0:
        raise RuntimeError(
            f"{split_name} merge 行数变化：{n0} -> {len(merged)}"
        )

    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched) > 0:
        show_cols = [
            c for c in merge_keys + ["task_id", "topic_id", "source_label"]
            if c in unmatched.columns
        ]
        raise ValueError(
            f"{split_name} 有 {len(unmatched)} 条 base 记录没有匹配 perturbation：\n"
            f"{unmatched[show_cols].head(20)}"
        )

    merged = merged.drop(columns=["_merge"])

    inconsistent = merged["label"].astype(int) != merged["perturb_label"].astype(int)
    if inconsistent.any():
        raise ValueError(
            f"{split_name} base 与 perturbation label 不一致：\n"
            f"{merged.loc[inconsistent, merge_keys + ['source_label', 'label', 'perturb_label']].head(20)}"
        )

    merged = merged.drop(columns=["perturb_label"])

    all_missing = merged[PERTURBATION_FEATURES].isna().all(axis=1)
    if all_missing.any():
        raise ValueError(
            f"{split_name} 有 {int(all_missing.sum())} 条记录 perturbation 特征全为空"
        )

    print("=" * 80)
    print(f"[{split_name}] merge mode:", strategy["mode"])
    print(f"[{split_name}] merge keys:", merge_keys)
    print(f"[{split_name}] merged rows:", len(merged))
    print(merged["source_label"].value_counts())

    return merged


def prepare_group_column(df, split_name):
    if "topic_id" not in df.columns:
        raise KeyError(
            f"{split_name} 没有 topic_id，无法做 grouped CV"
        )

    df = df.copy()
    df["topic_id"] = df["topic_id"].astype(str).str.strip()

    pair_table = (
        df.groupby(["topic_id", "source_label"])
        .size()
        .unstack(fill_value=0)
    )

    h = pair_table["human"] if "human" in pair_table.columns else 0
    m = pair_table["machine"] if "machine" in pair_table.columns else 0

    valid = (h == 1) & (m == 1)

    if (~valid).any():
        raise ValueError(
            f"{split_name} 存在非严格 human-machine paired topic：\n"
            f"{pair_table.loc[~valid].head(20)}"
        )

    print(f"[{split_name}] valid paired topics:", int(valid.sum()))
    return df


def validate_feature_columns(train_df, test_df):
    missing_train = [
        x for x in ALL_FEATURES
        if x not in train_df.columns
    ]

    missing_test = [
        x for x in ALL_FEATURES
        if x not in test_df.columns
    ]

    if missing_train or missing_test:
        raise KeyError(
            f"特征不完整：\n"
            f"train missing: {missing_train}\n"
            f"test missing: {missing_test}"
        )

    print("=" * 80)
    print("Structural path features:", len(STRUCTURAL_PATH_FEATURES))
    print("Revision path features:", len(REVISION_PATH_FEATURES))
    print("Probabilistic path features:", len(PROB_PATH_FEATURES))
    print("Total:", len(ALL_FEATURES))
    print("=" * 80)


# ============================================================
# 5. Preprocessing
# ============================================================

class PathPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit_transform(self, df, cols):
        x = df[cols].replace([np.inf, -np.inf], np.nan)
        x = self.imputer.fit_transform(x)
        x = self.scaler.fit_transform(x)
        return x.astype(np.float32)

    def transform(self, df, cols):
        x = df[cols].replace([np.inf, -np.inf], np.nan)
        x = self.imputer.transform(x)
        x = self.scaler.transform(x)
        return x.astype(np.float32)


class ThreePathDataset(Dataset):
    def __init__(self, xs, xr, xp, y):
        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.xr = torch.tensor(xr, dtype=torch.float32)
        self.xp = torch.tensor(xp, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.xs[idx], self.xr[idx], self.xp[idx], self.y[idx]


# ============================================================
# 6. Paper-style MHF model
# ============================================================

class PathEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=48, dropout=0.20):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PaperMHF(nn.Module):
    def __init__(
        self,
        structural_dim,
        revision_dim,
        prob_dim,
        hidden_dim=48,
        dropout=0.20,
        heads=4,
    ):
        super().__init__()

        self.structural_encoder = PathEncoder(
            structural_dim,
            hidden_dim,
            dropout,
        )

        self.revision_encoder = PathEncoder(
            revision_dim,
            hidden_dim,
            dropout,
        )

        self.prob_encoder = PathEncoder(
            prob_dim,
            hidden_dim,
            dropout,
        )

        self.path_embedding = nn.Parameter(
            torch.randn(3, hidden_dim) * 0.02
        )

        self.cross_path_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.path_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # auxiliary heads encourage each path to learn discriminative representation
        self.aux_structural = nn.Linear(hidden_dim, 1)
        self.aux_revision = nn.Linear(hidden_dim, 1)
        self.aux_prob = nn.Linear(hidden_dim, 1)

    def forward(self, xs, xr, xp):
        hs = self.structural_encoder(xs)
        hr = self.revision_encoder(xr)
        hp = self.prob_encoder(xp)

        h = torch.stack([hs, hr, hp], dim=1)
        h = h + self.path_embedding.unsqueeze(0)

        h_attn, _ = self.cross_path_attention(h, h, h)

        gate_logits = self.path_gate(h_attn).squeeze(-1)
        gate_weights = torch.softmax(gate_logits, dim=1)

        fused = torch.sum(
            h_attn * gate_weights.unsqueeze(-1),
            dim=1,
        )

        main_logit = self.classifier(fused).squeeze(-1)

        aux_logits = [
            self.aux_structural(h_attn[:, 0, :]).squeeze(-1),
            self.aux_revision(h_attn[:, 1, :]).squeeze(-1),
            self.aux_prob(h_attn[:, 2, :]).squeeze(-1),
        ]

        return main_logit, gate_weights, aux_logits


# ============================================================
# 7. Training / prediction
# ============================================================

def make_loader(df, preprocessors, fit, batch_size, shuffle):
    ps, pr, pp = preprocessors

    if fit:
        xs = ps.fit_transform(df, STRUCTURAL_PATH_FEATURES)
        xr = pr.fit_transform(df, REVISION_PATH_FEATURES)
        xp = pp.fit_transform(df, PROB_PATH_FEATURES)
    else:
        xs = ps.transform(df, STRUCTURAL_PATH_FEATURES)
        xr = pr.transform(df, REVISION_PATH_FEATURES)
        xp = pp.transform(df, PROB_PATH_FEATURES)

    y = df["label"].astype(int).to_numpy()

    ds = ThreePathDataset(xs, xr, xp, y)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def predict_model(model, loader, device):
    model.eval()

    probs = []
    ys = []
    gates = []

    with torch.no_grad():
        for xs, xr, xp, y in loader:
            xs = xs.to(device)
            xr = xr.to(device)
            xp = xp.to(device)

            logit, gate, _ = model(xs, xr, xp)
            prob = torch.sigmoid(logit)

            probs.append(prob.cpu().numpy())
            ys.append(y.numpy())
            gates.append(gate.cpu().numpy())

    return (
        np.concatenate(ys).astype(int),
        np.concatenate(probs),
        np.concatenate(gates),
    )


def split_train_val_by_group(df, cv_splits, random_state):
    y = df["label"].astype(int).to_numpy()
    groups = df["topic_id"].astype(str).to_numpy()

    sgkf = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    train_idx, val_idx = next(
        sgkf.split(df, y, groups)
    )

    return (
        df.iloc[train_idx].copy(),
        df.iloc[val_idx].copy(),
    )


def train_one_mhf(
    train_df,
    val_df,
    test_df,
    args,
    seed,
):
    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    ps = PathPreprocessor()
    pr = PathPreprocessor()
    pp = PathPreprocessor()

    train_loader = make_loader(
        train_df,
        (ps, pr, pp),
        fit=True,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        val_df,
        (ps, pr, pp),
        fit=False,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = make_loader(
        test_df,
        (ps, pr, pp),
        fit=False,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = PaperMHF(
        structural_dim=len(STRUCTURAL_PATH_FEATURES),
        revision_dim=len(REVISION_PATH_FEATURES),
        prob_dim=len(PROB_PATH_FEATURES),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        heads=args.heads,
    ).to(device)

    y_train = train_df["label"].astype(int).to_numpy()

    pos = max((y_train == 1).sum(), 1)
    neg = max((y_train == 0).sum(), 1)

    pos_weight = torch.tensor(
        [neg / pos],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        for xs, xr, xp, y in train_loader:
            xs = xs.to(device)
            xr = xr.to(device)
            xp = xp.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            main_logit, gate, aux_logits = model(xs, xr, xp)

            main_loss = criterion(main_logit, y)

            aux_loss = 0.0
            for aux_logit in aux_logits:
                aux_loss = aux_loss + criterion(aux_logit, y)

            aux_loss = aux_loss / len(aux_logits)

            loss = main_loss + args.aux_weight * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

        y_val, p_val, _ = predict_model(
            model,
            val_loader,
            device,
        )

        try:
            val_auc = roc_auc_score(
                y_val,
                p_val,
            )
        except Exception:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= args.patience:
            break

    model.load_state_dict(best_state)

    y_val, p_val, g_val = predict_model(
        model,
        val_loader,
        device,
    )

    y_test, p_test, g_test = predict_model(
        model,
        test_loader,
        device,
    )

    return {
        "model": model,
        "val_y": y_val,
        "val_prob": p_val,
        "val_gate": g_val,
        "test_y": y_test,
        "test_prob": p_test,
        "test_gate": g_test,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
    }


# ============================================================
# 8. OOF threshold
# ============================================================

def grouped_oof_predictions(train_df, args):
    y = train_df["label"].astype(int).to_numpy()
    groups = train_df["topic_id"].astype(str).to_numpy()

    sgkf = StratifiedGroupKFold(
        n_splits=args.cv_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    oof_prob = np.zeros(len(train_df), dtype=float)
    oof_gate = np.zeros((len(train_df), 3), dtype=float)

    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(train_df, y, groups),
        start=1,
    ):
        fold_train = train_df.iloc[tr_idx].copy()
        fold_val = train_df.iloc[va_idx].copy()

        payload = train_one_mhf(
            train_df=fold_train,
            val_df=fold_val,
            test_df=fold_val,
            args=args,
            seed=args.random_state + fold,
        )

        oof_prob[va_idx] = payload["test_prob"]
        oof_gate[va_idx, :] = payload["test_gate"]

        fold_auc = roc_auc_score(
            fold_val["label"].astype(int).to_numpy(),
            payload["test_prob"],
        )

        fold_aucs.append(fold_auc)

        print(
            f"[OOF fold {fold}] AUC={fold_auc:.4f} "
            f"best_val_auc={payload['best_val_auc']:.4f}"
        )

    oof_auc = roc_auc_score(y, oof_prob)

    return oof_prob, oof_gate, oof_auc, fold_aucs


# ============================================================
# 9. Evaluation
# ============================================================

def find_best_threshold(y_true, prob):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.2, 0.8, 121):
        pred = (prob >= t).astype(int)
        score = f1_score(
            y_true,
            pred,
            average="weighted",
        )

        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t, best_f1


def evaluate_predictions(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    )

    precision_w, recall_w, _, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="weighted",
        zero_division=0,
    )

    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "weighted_precision": precision_w,
        "weighted_recall": recall_w,
        "weighted_f1": f1_score(
            y_true,
            pred,
            average="weighted",
        ),
        "machine_precision": precision_m,
        "machine_recall": recall_m,
        "machine_f1": f1_m,
        "roc_auc": roc_auc_score(y_true, prob),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "pred": pred,
        "cm": cm,
    }


# ============================================================
# 10. Main
# ============================================================

def main():
    args = parse_args()

    set_seed(args.random_state)

    require_file(args.base_train_csv, "base train CSV")
    require_file(args.base_test_csv, "base test CSV")
    require_file(args.perturb_train_csv, "perturbation train CSV")
    require_file(args.perturb_test_csv, "perturbation test CSV")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_train_merged = out_dir / "train700_paper_mhf_three_path_merged.csv"
    output_test_merged = out_dir / "test200_paper_mhf_three_path_merged.csv"
    output_metrics = out_dir / "paper_mhf_three_path_metrics.csv"
    output_predictions = out_dir / "paper_mhf_three_path_predictions.csv"

    base_train = load_base_data(args.base_train_csv, "train700")
    base_test = load_base_data(args.base_test_csv, "test200")

    perturb_train = load_perturbation_data(args.perturb_train_csv, "train700")
    perturb_test = load_perturbation_data(args.perturb_test_csv, "test200")

    train_df = merge_base_and_perturbation(
        base_train,
        perturb_train,
        "train700",
    )

    test_df = merge_base_and_perturbation(
        base_test,
        perturb_test,
        "test200",
    )

    train_df = prepare_group_column(train_df, "train700")
    test_df = prepare_group_column(test_df, "test200")

    validate_feature_columns(train_df, test_df)

    train_df.to_csv(
        output_train_merged,
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        output_test_merged,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved merged train:", output_train_merged)
    print("Saved merged test:", output_test_merged)

    print("=" * 80)
    print("Train rows:", len(train_df))
    print(train_df["source_label"].value_counts())
    print()
    print("Test rows:", len(test_df))
    print(test_df["source_label"].value_counts())
    print()
    print("Train topics:", train_df["topic_id"].nunique())
    print("Test topics:", test_df["topic_id"].nunique())
    print("=" * 80)

    # 1. grouped OOF probability for threshold selection
    print("[STEP 1] grouped OOF prediction for threshold selection")
    oof_prob, oof_gate, oof_auc, fold_aucs = grouped_oof_predictions(
        train_df,
        args,
    )

    y_train = train_df["label"].astype(int).to_numpy()

    best_threshold, best_oof_f1 = find_best_threshold(
        y_train,
        oof_prob,
    )

    print("=" * 80)
    print("Grouped OOF AUC:", round(oof_auc, 5))
    print("Fold AUCs:", [round(x, 5) for x in fold_aucs])
    print("Best threshold from grouped OOF:", round(best_threshold, 4))
    print("Best grouped OOF weighted F1:", round(best_oof_f1, 5))
    print("OOF average gate:",
          np.round(oof_gate.mean(axis=0), 4))

    # 2. final model trained from train split with grouped validation
    print("=" * 80)
    print("[STEP 2] final MHF model for held-out test200")

    final_train, final_val = split_train_val_by_group(
        train_df,
        args.cv_splits,
        args.random_state,
    )

    final_payload = train_one_mhf(
        train_df=final_train,
        val_df=final_val,
        test_df=test_df,
        args=args,
        seed=args.random_state,
    )

    y_test = test_df["label"].astype(int).to_numpy()
    p_test = final_payload["test_prob"]
    gate_test = final_payload["test_gate"]

    metrics_05 = evaluate_predictions(
        y_test,
        p_test,
        threshold=0.5,
    )

    metrics_tuned = evaluate_predictions(
        y_test,
        p_test,
        threshold=best_threshold,
    )

    rows = []

    base_row = {
        "model": "PaperMHF_three_path",
        "fusion": "path_encoder_cross_attention_adaptive_gate",
        "n_structural_features": len(STRUCTURAL_PATH_FEATURES),
        "n_revision_features": len(REVISION_PATH_FEATURES),
        "n_prob_features": len(PROB_PATH_FEATURES),
        "grouped_oof_auc": oof_auc,
        "grouped_oof_best_weighted_f1": best_oof_f1,
        "best_threshold_from_grouped_oof": best_threshold,
        "final_best_val_auc": final_payload["best_val_auc"],
        "final_best_epoch": final_payload["best_epoch"],
        "gate_structural": float(gate_test[:, 0].mean()),
        "gate_revision": float(gate_test[:, 1].mean()),
        "gate_prob": float(gate_test[:, 2].mean()),
    }

    for threshold_type, metrics in [
        ("0.5", metrics_05),
        ("grouped_oof_tuned", metrics_tuned),
    ]:
        row = {
            **base_row,
            "threshold_type": threshold_type,
            "threshold": metrics["threshold"],
            "accuracy": metrics["accuracy"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "machine_precision": metrics["machine_precision"],
            "machine_recall": metrics["machine_recall"],
            "machine_f1": metrics["machine_f1"],
            "roc_auc": metrics["roc_auc"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(
        output_metrics,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("Final paper-style MHF three-path result:")
    print(
        results_df[[
            "model",
            "threshold_type",
            "threshold",
            "accuracy",
            "weighted_f1",
            "machine_f1",
            "roc_auc",
            "grouped_oof_auc",
            "final_best_val_auc",
            "gate_structural",
            "gate_revision",
            "gate_prob",
            "tn",
            "fp",
            "fn",
            "tp",
        ]].to_string(index=False)
    )

    print()
    print("[TEST classification report @ grouped OOF threshold]")
    print(
        classification_report(
            y_test,
            metrics_tuned["pred"],
            labels=[0, 1],
            target_names=["human", "machine"],
            digits=4,
            zero_division=0,
        )
    )

    pred_cols = [
        c for c in [
            "task_id",
            "sample_id",
            "topic_id",
            "source_label",
            "label",
        ]
        if c in test_df.columns
    ]

    pred_df = test_df[pred_cols].copy()
    pred_df["prob_machine"] = p_test
    pred_df["pred_05"] = metrics_05["pred"]
    pred_df["pred_grouped_oof_tuned"] = metrics_tuned["pred"]
    pred_df["threshold_grouped_oof"] = best_threshold
    pred_df["gate_structural"] = gate_test[:, 0]
    pred_df["gate_revision"] = gate_test[:, 1]
    pred_df["gate_prob"] = gate_test[:, 2]

    pred_df.to_csv(
        output_predictions,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Saved metrics to:", output_metrics)
    print("Saved predictions to:", output_predictions)


if __name__ == "__main__":
    main()
