import json
import math
from collections import Counter
import spacy
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_curve
import matplotlib.pyplot as plt

# 配置
SAMPLED_FILE    = "/root/autodl-tmp/9/m4_cleaned_dataset.jsonl"   # 你的 JSONL 数据
DEP_DIST_FILE   = "/root/autodl-tmp/9/dependency_label_distribution_full_merged.csv"
OUTPUT_FILE     = "/root/autodl-tmp/9/static_metrics_5feats.jsonl"
SPACY_MODEL     = "zh_core_web_sm"

# 工具函数
def compute_ttr(doc):
    tokens = [t.text for t in doc if not t.is_punct and not t.is_space]
    total = len(tokens)
    return len(set(tokens)) / total if total > 0 else 0.0

def compute_mean_arclen(doc):
    lengths = [abs(tok.head.i - tok.i) for tok in doc if tok.head != tok]
    return sum(lengths) / len(lengths) if lengths else 0.0

def extract_features(doc, P_h):
    # KL
    deps_x = [tok.dep_ for tok in doc]
    cx = Counter(deps_x)
    total_x = sum(cx.values()) or 1
    KL = 0.0
    for dep, ph in P_h.items():
        px = cx.get(dep, 0) / total_x
        if px > 0 and ph > 0:
            KL += px * math.log(px / ph)

    # 其他特征
    LenToken = len([t for t in doc if not t.is_punct and not t.is_space])
    MeanWordLen = sum(len(t.text) for t in doc if not t.is_punct and not t.is_space) / (LenToken or 1)
    StopwordRatio = sum(1 for t in doc if t.is_stop) / (LenToken or 1)
    MeanArcLen = compute_mean_arclen(doc)

    return {
        "SDV1_TTR": compute_ttr(doc),
        "SDV2_KL": KL,
        "SDV3_MeanWordLen": MeanWordLen,
        "SDV4_MeanArcLen": MeanArcLen,
        "SDV5_StopwordRatio": StopwordRatio
    }

# MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def main():
    print("Loading spaCy model...")
    nlp = spacy.load(SPACY_MODEL, disable=["ner","senter","attribute_ruler"])

    print("Loading human dependency distribution...")
    df_dep = pd.read_csv(DEP_DIST_FILE)
    total_h = df_dep["Count"].sum() or 1
    P_h = {row["DepLabel"]: row["Count"]/total_h for _, row in df_dep.iterrows()}

    print("Reading JSONL dataset...")
    with open(SAMPLED_FILE, "r", encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin if line.strip()]

    # 自动检测 text 和 label 字段
    sample = records[0]
    text_key = "text" if "text" in sample else "content" if "content" in sample else list(sample.keys())[0]
    label_key = "label" if "label" in sample else "class" if "class" in sample else list(sample.keys())[1]
    print(f"Detected text field = {text_key}, label field = {label_key}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for rec in tqdm(records, desc="  sampled data"):
            txt   = rec.get(text_key, "").strip()
            label = rec.get(label_key, "Unknown")
            if not txt:
                continue
            doc = nlp(txt)
            feats = extract_features(doc, P_h)
            feats["label"] = label
            out_f.write(json.dumps(feats, ensure_ascii=False) + "\n")

    print(f"Done! Features saved to {OUTPUT_FILE}")

def analyze_results():
    print("Analyzing results with MLP...")
    df = pd.read_json(OUTPUT_FILE, lines=True)

    # 二分类: Human=1, AI=0
    df["y_true"] = df["label"].map({"Human": 1, "AI": 0})

    feature_cols = [c for c in df.columns if c not in ["label", "y_true"]]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df["y_true"].values, dtype=torch.float32).unsqueeze(1)

    # 定义模型
    model = MLP(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    for epoch in range(20):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    print(f"Final training loss = {loss.item():.4f}")

    # 评估
    with torch.no_grad():
        y_score = model(X).numpy().flatten()
    y_true = y.numpy().flatten()

    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, (y_score>0.5).astype(int))
    f1 = f1_score(y_true, (y_score>0.5).astype(int))
    prec = precision_score(y_true, (y_score>0.5).astype(int))
    rec = recall_score(y_true, (y_score>0.5).astype(int))

    print(f"MLP AUC = {auc:.3f}, Accuracy = {acc:.3f}, F1 = {f1:.3f}, Precision = {prec:.3f}, Recall = {rec:.3f}")
    print("\nClassification Report:\n", classification_report(y_true, (y_score>0.5).astype(int)))

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"MLP (AUC = {auc:.3f})")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - MLP (5 SDV Features)")
    plt.legend()
    plt.savefig("mlp_sdv5_roc.png")
    plt.close()
    print("ROC 曲线已保存: mlp_sdv5_roc.png")

if __name__ == "__main__":
    main()
    analyze_results()
