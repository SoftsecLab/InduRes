
import json
import math
from collections import Counter

import spacy
import pandas as pd
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# -------------------------
# 配置
# -------------------------
DATASETS = {
    "Human":   "/home/chy/tiaoshi/merged_human.jsonl",
    "QianWen": "/home/chy/tiaoshi/11qianwen_filled.jsonl",
    "WenXin":  "/home/chy/tiaoshi/wenxin_machine_wenxin_8.jsonl",
    "GLM":     "/home/chy/tiaoshi/glm4_machine_rewrites.jsonl",
    "AiYi":    "/home/chy/tiaoshi/rewrite_Yi.jsonl",
}

SPACY_MODEL = "zh_core_web_sm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 200
BATCH_SIZE = 512
LR = 1e-3


# =======================
# 1. SDV 特征函数
# =======================
def compute_ttr(doc):
    """Token Type Ratio"""
    tokens = [t.text for t in doc if not t.is_punct and not t.is_space]
    total = len(tokens)
    return len(set(tokens)) / total if total > 0 else 0.0


def compute_mean_arclen(doc):
    """平均依存弧长"""
    lengths = [abs(tok.head.i - tok.i) for tok in doc if tok.head != tok]
    return sum(lengths) / len(lengths) if lengths else 0.0


def extract_sdv_features(doc, P_h):

    # ---- KL: 依存标签分布与 Human 的 KL 散度 ----
    deps_x = [tok.dep_ for tok in doc]
    cx = Counter(deps_x)
    total_x = sum(cx.values()) or 1

    KL = 0.0
    for dep, ph in P_h.items():
        px = cx.get(dep, 0) / total_x
        if px > 0 and ph > 0:
            KL += px * math.log(px / ph)

    # ---- 其他基本特征 ----
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    LenToken = len(tokens)
    MeanWordLen = sum(len(t.text) for t in tokens) / (LenToken or 1)
    StopwordRatio = sum(1 for t in tokens if t.is_stop) / (LenToken or 1)
    MeanArcLen = compute_mean_arclen(doc)

    return [
        compute_ttr(doc),
        KL,
        MeanWordLen,
        MeanArcLen,
        StopwordRatio
    ]


# =======================
# 2. 加载数据 & 计算特征
# =======================
print("\nLoading spaCy model...")
nlp = spacy.load(SPACY_MODEL, disable=["ner", "senter", "attribute_ruler"])

# ---- 先统计 Human 依存分布 P_h(dep) ----
print("\nComputing Human dependency distribution...")
dep_counter = Counter()
total_h_tokens = 0

human_path = DATASETS["Human"]
with open(human_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="[Human DepDist]"):
        obj = json.loads(line)
        for txt in obj.get("rewrites", []):
            doc = nlp(txt)
            for tok in doc:
                dep_counter[tok.dep_] += 1
                total_h_tokens += 1

P_h = {dep: count / total_h_tokens for dep, count in dep_counter.items()}

# ---- 正式提取所有 SDV 特征 ----
rows = []
print("\nExtracting SDV features...")
for label_name, path in DATASETS.items():
    label = 0 if label_name == "Human" else 1  # Human=0, Machine=1

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"[{label_name}]"):
            obj = json.loads(line)
            for txt in obj.get("rewrites", []):
                txt = txt.strip()
                if not txt:
                    continue
                doc = nlp(txt)
                feats = extract_sdv_features(doc, P_h)  # 5 维

                rows.append(feats + [label])

df = pd.DataFrame(rows, columns=[
    "TTR", "KL", "MeanWordLen", "MeanArcLen", "StopwordRatio", "label"
])

print("\nTotal samples:", len(df))
print(df["label"].value_counts())


# =======================
# 3. Train / Test 划分
# =======================
X = df[["TTR", "KL", "MeanWordLen", "MeanArcLen", "StopwordRatio"]].values
y = df["label"].values  # 0=Human, 1=Machine

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_test_t  = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)


# =======================
# 4. 模型：ResMLP + FocalLoss
# =======================
class ResBlock(nn.Module):
    """简单一层残差块：Linear -> LayerNorm -> GELU -> Dropout + 残差"""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc(x)
        h = self.ln(h)
        h = self.act(h)
        h = self.dropout(h)
        return x + h  # 残差


class SDV_ResMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.block1 = ResBlock(hidden_dim, dropout=0.2)
        self.block2 = ResBlock(hidden_dim, dropout=0.2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 输出概率
        )

    def forward(self, x):
        h = self.input(x)
        h = self.block1(h)
        h = self.block2(h)
        out = self.head(h)
        return out


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (with prob input)
    inputs: p in (0,1), targets in {0,1}
    """
    def __init__(self, alpha=0.5, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        # inputs: [B,1], targets: [B,1]
        p = inputs.clamp(self.eps, 1.0 - self.eps)
        y = targets

        # BCE 部分
        bce = - (self.alpha * y * torch.log(p) +
                 (1 - self.alpha) * (1 - y) * torch.log(1 - p))

        # pt = p  (if y=1) or 1-p (if y=0)
        pt = y * p + (1 - y) * (1 - p)

        loss = (1 - pt) ** self.gamma * bce
        return loss.mean()


model = SDV_ResMLP(input_dim=5, hidden_dim=64).to(DEVICE)
criterion = FocalLoss(alpha=0.5, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# =======================
# 5. 训练
# =======================
print("\nTraining SDV ResMLP + FocalLoss...")

N = X_train_t.size(0)
indices = np.arange(N)

for epoch in range(EPOCHS):
    model.train()
    np.random.shuffle(indices)
    total_loss = 0.0

    for start in range(0, N, BATCH_SIZE):
        end = start + BATCH_SIZE
        idx = indices[start:end]

        x = X_train_t[idx]
        yb = y_train_t[idx]

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)

    scheduler.step()

    if (epoch + 1) % 20 == 0 or epoch == 0:
        avg_loss = total_loss / N
        print(f"Epoch {epoch+1}/{EPOCHS}, TrainLoss = {avg_loss:.4f}")



model.eval()
with torch.no_grad():
    y_score = model(X_test_t).cpu().numpy().flatten()

y_pred = (y_score > 0.5).astype(int)
y_true = y_test_t.cpu().numpy().flatten()

print("\n=== SDV ResMLP + FocalLoss Report ===")
print(classification_report(y_true, y_pred, target_names=["Human", "Machine"]))
print("Accuracy :", accuracy_score(y_true, y_pred))
print("AUC      :", roc_auc_score(y_true, y_score))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1       :", f1_score(y_true, y_pred))
