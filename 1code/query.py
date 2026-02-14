# -*- coding: utf-8 -*-
"""
Hybrid 3-path Fusion (Improved + Multi-Head + FocalLoss / 可直接运行)

保持指标公式不变，只增强融合与损失：
- Path1: SRM / PathVar / EDC / PDJ           (4d) -> 与 Path2 Multi-Head Cross Attention 交叉融合
- Path2: d_freq / d_trans / A / B / C       (5d) -> 与 Path1 Multi-Head Cross Attention 交叉融合
- Path3: SDV 静态特征 5d                     (5d) -> 更强 SDVEncoder（容量↑）

融合结构：
1) MultiHeadCrossPathAttention(Path1, Path2) -> h12 -> logits12
2) SDVEncoder(Path3) -> h3 -> logits3
3) Late-Fusion 门控（embedding级别）:
    gate = sigmoid(MLP([h12,h3]))
    h = gate*h12 + (1-gate)*h3
    logits = FC(h)

改进点：
(1) Path1+2 交叉融合用 Multi-Head Attention（NUM_HEADS 可调）
(2) Path1+2 cross 输出后 Dropout(0.4)
(3) Path3 SDVEncoder 更深更大(hidden_dim=64)
(4) Late fusion gate + 分类 head 前 LayerNorm
(5) FocalLoss 处理不均衡（alpha 可调，默认偏向 Human）

Author: you + ChatGPT
"""

import json
import math
from collections import Counter

import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR


# ================================
# 0. 配置
# ================================
DATASETS = {
    "Human":   "/home/chy/tiaoshi/merged_human.jsonl",
    "QianWen": "/home/chy/tiaoshi/11qianwen_filled.jsonl",
    "WenXin":  "/home/chy/tiaoshi/wenxin_machine_wenxin_8.jsonl",
    "GLM":     "/home/chy/tiaoshi/glm4_machine_rewrites.jsonl",
    "AiYi":    "/home/chy/tiaoshi/rewrite_Yi.jsonl",
}

GLOBALS_PATH = "/home/chy/tiaoshi/syntax_globals.joblib"
FUSED_CSV    = "/home/chy/new/fused_features_3path_hybrid_mha_focal.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型与训练
HIDDEN_DIM = 64
NUM_HEADS  = 4          # ✅ Multi-Head 个数（可改 2/4/8）
EPOCHS     = 200
TEST_SIZE  = 0.2
BATCH_SIZE = 512
LR         = 1e-3
SEED       = 42
MAX_D_PDJ  = 20

# FocalLoss
FOCAL_GAMMA = 2.0
# alpha: [Human, Machine]，Human 权重略大以提升召回
FOCAL_ALPHA = torch.tensor([0.65, 0.35], device=DEVICE)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================
# 1. 加载 spaCy 模型
# ================================
try:
    nlp = spacy.load("zh_core_web_sm", disable=["ner", "senter", "attribute_ruler"])
except OSError:
    raise OSError("未找到 zh_core_web_sm，请先运行：python -m spacy download zh_core_web_sm")


# ================================
# 2. 加载 globals（Path2 用）
# ================================
globals_dict = joblib.load(GLOBALS_PATH)

hf = globals_dict["hf"]
mf = globals_dict["mf"]
Ph = globals_dict["Ph"]
Pm = globals_dict["Pm"]
all_deps  = globals_dict["all_deps"]
all_pairs = globals_dict["all_pairs"]
w_components = globals_dict["w_components"]

# 论文里 KL 的 human 参考分布 P_h
P_h = dict(hf)


# ================================
# 3. 通用函数
# ================================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line.strip())
            source = obj.get("source", "")
            rewrites = obj.get("rewrites", None) or obj.get("machine_rewrites", [])
            data.append({"source": source, "rewrites": rewrites})
    return data


def levenshtein_distance(a, b):
    len_a, len_b = len(a), len(b)
    dp = [[0]*(len_b+1) for _ in range(len_a+1)]
    for i in range(len_a+1): dp[i][0] = i
    for j in range(len_b+1): dp[0][j] = j
    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[len_a][len_b]


def parse_sentence(sent):
    doc = nlp(sent)
    dep_labels = []
    dep_edges = set()
    nodes = len(doc)

    for tok in doc:
        dep_labels.append(tok.dep_)
        head_idx = tok.head.i if tok.head is not None else tok.i
        dep_edges.add((head_idx, tok.i, tok.dep_))

    def get_depth(i):
        depth, cur = 0, i
        while True:
            head = doc[cur].head.i
            if head == cur:
                break
            depth += 1
            cur = head
            if depth > nodes:
                break
        return depth

    max_depth = 1
    for i in range(nodes):
        max_depth = max(max_depth, get_depth(i))

    return {
        "dep_labels": dep_labels,
        "dep_edges": dep_edges,
        "nodes": nodes,
        "depth": max_depth,
    }


# ================================
# 4. Path1 指标（完全按论文公式）
# ================================
def compute_SRM(src_parse, tgt_parse):
    seq_src = src_parse["dep_labels"]
    seq_tgt = tgt_parse["dep_labels"]
    dist = levenshtein_distance(seq_src, seq_tgt)
    length = len(seq_src) if seq_src else 1
    return dist / length


def compute_EDC(src_parse, rw_parse):
    Ei = src_parse["dep_edges"]
    Ej = rw_parse["dep_edges"]
    inter = len(Ei & Ej)
    if not Ei and not Ej:
        return 0.0
    prec = inter / (len(Ej) if Ej else 1)
    rec  = inter / (len(Ei) if Ei else 1)
    if prec + rec == 0:
        return 1.0
    F1 = 2 * prec * rec / (prec + rec)
    return 1.0 - F1


def compute_PDJ(src_parse, rw_parse, max_d=MAX_D_PDJ):
    def hist_norm(depths):
        H = [0]*(max_d+1)
        for d in depths:
            H[min(d, max_d)] += 1
        s = sum(H) or 1
        return [h/s for h in H]
    hs = hist_norm([src_parse["depth"]])
    hr = hist_norm([rw_parse["depth"]])
    return sum(abs(a-b) for a,b in zip(hs, hr))


def compute_PathVar_per_rewrite(parse_list):
    n = len(parse_list)
    if n < 2:
        return [0.0]*n
    out = []
    for i in range(n):
        Ei = parse_list[i]["dep_edges"]
        total, cnt = 0.0, 0
        for j in range(n):
            if i==j: continue
            Ej = parse_list[j]["dep_edges"]
            total += len(Ei.symmetric_difference(Ej))
            cnt += 1
        out.append(total/cnt if cnt else 0.0)
    return out


# ================================
# 5. Path2 指标（完全按论文公式）
# ================================
def extract_path2_features(text):
    doc = nlp(text)

    # dep freq
    cnt_f, tot_f = Counter(), 0
    for tok in doc:
        cnt_f[tok.dep_] += 1
        tot_f += 1
    f_text = {d: cnt_f[d]/tot_f if tot_f else 0 for d in all_deps}

    # dep transition
    trans_txt, tot_t = Counter(), Counter()
    for sent in doc.sents:
        deps = [tok.dep_ for tok in sent]
        for a,b in zip(deps, deps[1:]):
            trans_txt[(a,b)] += 1
            tot_t[a] += 1
    P_text = {
        p: (trans_txt[p]/tot_t[p[0]] if tot_t[p[0]]>0 else 0)
        for p in all_pairs
    }

    # adaptive A/B/C
    if len(doc) > 0:
        A,B,C = w_components.get(doc[0].text, (0.0,0.0,0.0))
    else:
        A,B,C = 0.0,0.0,0.0

    d_freq = sum(abs(f_text[d]-mf.get(d,0)) for d in all_deps) \
           - sum(abs(f_text[d]-hf.get(d,0)) for d in all_deps)

    d_trans = sum(abs(P_text[p]-Pm.get(p,0)) for p in all_pairs) \
            - sum(abs(P_text[p]-Ph.get(p,0)) for p in all_pairs)

    return d_freq, d_trans, A, B, C


# ================================
# 6. Path3 SDV 指标（完全按论文公式）
# ================================
def compute_ttr(doc):
    tokens = [t.text for t in doc if not t.is_punct and not t.is_space]
    total = len(tokens)
    return len(set(tokens))/total if total>0 else 0.0


def compute_mean_arclen(doc):
    lengths = [abs(tok.head.i - tok.i) for tok in doc if tok.head != tok]
    return sum(lengths)/len(lengths) if lengths else 0.0


def extract_sdv_features(doc, P_h):
    deps_x = [tok.dep_ for tok in doc]
    cx = Counter(deps_x)
    total_x = sum(cx.values()) or 1

    KL = 0.0
    for dep, ph in P_h.items():
        px = cx.get(dep, 0) / total_x
        if px > 0 and ph > 0:
            KL += px * math.log(px / ph)

    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    LenToken = len(tokens)
    MeanWordLen = sum(len(t.text) for t in tokens) / (LenToken or 1)
    StopwordRatio = sum(1 for t in tokens if t.is_stop) / (LenToken or 1)
    MeanArcLen = compute_mean_arclen(doc)
    TTR = compute_ttr(doc)

    return TTR, KL, MeanWordLen, MeanArcLen, StopwordRatio


# ================================
# 7. 提取三路径特征
# ================================
rows = []
print("\n=== [Step1] Extracting 3-path features (Hybrid MHA + Focal) ===")

for model_name, path in DATASETS.items():
    label = 0 if model_name=="Human" else 1
    data_list = load_jsonl(path)

    for entry in tqdm(data_list, desc=model_name, unit="source"):
        src = entry["source"]
        rws = entry["rewrites"]
        if not rws:
            continue

        src_parse = parse_sentence(src)
        rw_parses = [parse_sentence(r) for r in rws]
        pathvars = compute_PathVar_per_rewrite(rw_parses)

        for rw_text, rw_parse, pv in zip(rws, rw_parses, pathvars):
            SRM  = compute_SRM(src_parse, rw_parse)
            EDC  = compute_EDC(src_parse, rw_parse)
            PDJ  = compute_PDJ(src_parse, rw_parse)
            PathVar = pv

            d_freq, d_trans, A, B, C_val = extract_path2_features(rw_text)

            doc_rw = nlp(rw_text)
            TTR, KL, MWL, MAL, StopRatio = extract_sdv_features(doc_rw, P_h)

            rows.append({
                "SRM": SRM, "PathVar": PathVar, "EDC": EDC, "PDJ": PDJ,
                "d_freq": d_freq, "d_trans": d_trans, "A": A, "B": B, "C": C_val,
                "TTR": TTR, "KL": KL, "MeanWordLen": MWL,
                "MeanArcLen": MAL, "StopwordRatio": StopRatio,
                "label": label
            })

df = pd.DataFrame(rows)
df.to_csv(FUSED_CSV, index=False)
print(f"\n✅ Saved fused features to: {FUSED_CSV}")
print("Total samples:", len(df))


# ================================
# 8. 三路径数据切分 + 标准化
# ================================
X1 = df[["SRM","PathVar","EDC","PDJ"]].values
X2 = df[["d_freq","d_trans","A","B","C"]].values
X3 = df[["TTR","KL","MeanWordLen","MeanArcLen","StopwordRatio"]].values
y  = df["label"].values

X1_tr, X1_te, X2_tr, X2_te, X3_tr, X3_te, y_tr, y_te = train_test_split(
    X1, X2, X3, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

scaler1, scaler2, scaler3 = StandardScaler(), StandardScaler(), StandardScaler()
X1_tr = scaler1.fit_transform(X1_tr); X1_te = scaler1.transform(X1_te)
X2_tr = scaler2.fit_transform(X2_tr); X2_te = scaler2.transform(X2_te)
X3_tr = scaler3.fit_transform(X3_tr); X3_te = scaler3.transform(X3_te)

X1_tr_t = torch.tensor(X1_tr, dtype=torch.float32).to(DEVICE)
X2_tr_t = torch.tensor(X2_tr, dtype=torch.float32).to(DEVICE)
X3_tr_t = torch.tensor(X3_tr, dtype=torch.float32).to(DEVICE)
y_tr_t  = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)

X1_te_t = torch.tensor(X1_te, dtype=torch.float32).to(DEVICE)
X2_te_t = torch.tensor(X2_te, dtype=torch.float32).to(DEVICE)
X3_te_t = torch.tensor(X3_te, dtype=torch.float32).to(DEVICE)
y_te_t  = torch.tensor(y_te, dtype=torch.long).to(DEVICE)


# ================================
# 9. Focal Loss
# ================================
class FocalLoss(nn.Module):
    """
    多类 Focal Loss：
    FL = alpha_t * (1-pt)^gamma * CE
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # Tensor shape [C]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)  # pt = softmax prob of true class
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * (1 - pt) ** self.gamma * ce
        else:
            loss = (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ================================
# 10. Multi-Head Cross Attention
# ================================
class MultiHeadCrossPathAttention(nn.Module):
    """
    Path1 + Path2 多头交叉融合：
    - 先把 path1, path2 投到 hidden_dim
    - 用 MultiheadAttention 做：
        Hp = self_attn(path1) + cross_attn(path1 <- path2)
        Hr = self_attn(path2) + cross_attn(path2 <- path1)
    - 再按你原来做 alpha/beta 门控合并

    注意：这里把每条路径当作 seq_len=1 的 token，
    多头主要提供多子空间投影与稳定性（实验上对小维度特征更稳）。
    """
    def __init__(self, dim1, dim2, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)

        self.self1  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self2  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.alpha_layer = nn.Linear(hidden_dim, 1)
        self.beta_layer  = nn.Linear(hidden_dim, 1)

    def forward(self, x1, x2):
        # [B, H]
        h1 = self.proj1(x1)
        h2 = self.proj2(x2)

        # 变成 [B, 1, H]
        h1s = h1.unsqueeze(1)
        h2s = h2.unsqueeze(1)

        # path1: self + cross
        att11, _ = self.self1(h1s, h1s, h1s)          # [B,1,H]
        att12, _ = self.cross1(h1s, h2s, h2s)        # [B,1,H]
        Hp = (att11 + att12).squeeze(1)              # [B,H]

        # path2: self + cross
        att22, _ = self.self2(h2s, h2s, h2s)
        att21, _ = self.cross2(h2s, h1s, h1s)
        Hr = (att22 + att21).squeeze(1)

        alpha = torch.sigmoid(self.alpha_layer(Hp))
        beta  = torch.sigmoid(self.beta_layer(Hr))

        H12 = alpha * Hp + beta * Hr
        return H12


class Cross2PathNet(nn.Module):
    """Path1+2 多头交叉融合后 -> logits12"""
    def __init__(self, dim1, dim2, hidden_dim, num_heads=4):
        super().__init__()
        self.att = MultiHeadCrossPathAttention(dim1, dim2, hidden_dim, num_heads=num_heads)
        self.ln  = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.4)
        self.fc  = nn.Linear(hidden_dim, 2)

    def forward(self, x1, x2):
        h = self.att(x1, x2)
        h = self.ln(h)
        h = self.drop(h)
        logits12 = self.fc(h)
        return logits12, h


class SDVEncoder(nn.Module):
    """增强 Path3 SDV Encoder（保持指标公式不变）"""
    def __init__(self, in_dim=5, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h = self.net(x)
        logits3 = self.head(h)
        return logits3, h


class HybridFusion3PathNet_MHA_Focal(nn.Module):
    """
    最终 Improved Hybrid + MHA:
    - logits12 = MultiHeadCross(Path1,Path2)
    - logits3  = SDVEncoder(Path3)
    - 门控 late fusion 在 embedding 级别做
    """
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.cross12 = Cross2PathNet(4, 5, hidden_dim, num_heads=num_heads)
        self.sdv     = SDVEncoder(5, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x1, x2, x3):
        logits12, h12 = self.cross12(x1, x2)
        logits3,  h3  = self.sdv(x3)

        g = torch.sigmoid(self.gate(torch.cat([h12, h3], dim=1)))
        h = g*h12 + (1-g)*h3

        h = self.ln(h)
        fused_logits = self.fc(h)
        return fused_logits, logits12, logits3, g


model = HybridFusion3PathNet_MHA_Focal(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
scaler = GradScaler()

def autocast_ctx():
    if DEVICE.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    else:
        from contextlib import nullcontext
        return nullcontext()


# ================================
# 11. 训练
# ================================
print("\n=== [Step2] Training Improved Hybrid 3-path Net (MHA + FocalLoss) ===")
N = X1_tr_t.size(0)
indices = np.arange(N)

for epoch in range(EPOCHS):
    model.train()
    np.random.shuffle(indices)
    total_loss = 0.0

    for start in range(0, N, BATCH_SIZE):
        end = start + BATCH_SIZE
        idx = indices[start:end]

        bX1 = X1_tr_t[idx]
        bX2 = X2_tr_t[idx]
        bX3 = X3_tr_t[idx]
        by  = y_tr_t[idx]

        optimizer.zero_grad()
        with autocast_ctx():
            fused_logits, _, _, g = model(bX1, bX2, bX3)
            loss = criterion(fused_logits, by)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(idx)

    scheduler.step()

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss/N:.4f} | gate_mean={g.mean().item():.3f}")


# ================================
# 12. 测试评估
# ================================
model.eval()
with torch.no_grad():
    fused_logits, logits12, logits3, g = model(X1_te_t, X2_te_t, X3_te_t)
    probs  = torch.softmax(fused_logits, dim=1)[:,1]
    preds  = torch.argmax(fused_logits, dim=1)

y_true = y_te_t.cpu().numpy()
y_pred = preds.cpu().numpy()
y_prob = probs.cpu().numpy()

print("\n=== [Step3] Test Performance (Improved Hybrid + MHA + FocalLoss) ===")
print(classification_report(y_true, y_pred, target_names=["Human","Machine"]))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("ROC AUC :", roc_auc_score(y_true, y_prob))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nGate mean on test:", g.mean().item())
