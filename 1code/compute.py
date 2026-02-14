# -*- coding: utf-8 -*-
import json
import math
import os
import csv
from collections import Counter
import spacy
import pandas as pd
from tqdm import tqdm

# ---------- 1. 加载 spaCy 中文模型 ----------
try:
    nlp = spacy.load("zh_core_web_sm")
except OSError:
    raise OSError(
        "未找到 spaCy 中文模型，请先运行：\n"
        "  python -m spacy download zh_core_web_sm"
    )

# ---------- 2. 定义辅助函数 ----------
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
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[len_a][len_b]

def parse_sentence(sent):
    doc = nlp(sent)
    dep_labels = []
    dep_edges = set()
    nodes = len(doc)
    children_map = {i: [] for i in range(nodes)}

    for token in doc:
        idx = token.i
        dep_labels.append(token.dep_)
        head_idx = token.head.i if token.head is not None else token.i
        dep_edges.add((head_idx, idx, token.dep_))
        if token.i != head_idx:
            children_map[head_idx].append(idx)

    def get_depth(i):
        depth = 0
        cur = i
        while True:
            head = doc[cur].head.i
            if head == cur:
                break
            depth += 1
            cur = head
            if depth > nodes:
                break
        return depth

    max_depth = 0
    for i in range(nodes):
        d = get_depth(i)
        if d > max_depth:
            max_depth = d

    branch_count = sum(1 for chs in children_map.values() if len(chs) > 1)
    label_counts = Counter(dep_labels)

    return {
        "dep_labels": dep_labels,
        "dep_edges": dep_edges,
        "nodes": nodes,
        "depth": max_depth if max_depth > 0 else 1,
        "branch_count": branch_count,
        "label_counts": label_counts
    }

def compute_SRM(src_parse, tgt_parse):
    seq_src = src_parse["dep_labels"]
    seq_tgt = tgt_parse["dep_labels"]
    dist = levenshtein_distance(seq_src, seq_tgt)
    length = len(seq_src) if len(seq_src) > 0 else 1
    return dist / length

# ===== 改进版 PathVar =====
def compute_path_variability_weighted(parse_list):
    n = len(parse_list)
    if n < 2:
        return 0.0
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges_i = parse_list[i]["dep_edges"]
            edges_j = parse_list[j]["dep_edges"]
            sym_diff = edges_i.symmetric_difference(edges_j)
            weighted = 0
            for (h, t, dep) in sym_diff:
                depth = min(parse_list[i]["depth"], parse_list[j]["depth"])
                weighted += 1.0 / (1 + depth)
            total += weighted
            count += 1
    avg_nodes = sum(p["nodes"] for p in parse_list) / n
    return (total / count) / (avg_nodes if avg_nodes > 0 else 1)

# ===== 新指标（差异型） =====
def compute_EDC(src_parse, rw_parse):
    """Edge Delta Core: 源句 vs 改写句核心依存边差异 (1 - F1)"""
    Ei = src_parse["dep_edges"]
    Ej = rw_parse["dep_edges"]
    inter = len(Ei & Ej)
    if len(Ei) == 0 and len(Ej) == 0:
        return 0.0
    prec = inter / (len(Ej) if Ej else 1)
    rec  = inter / (len(Ei) if Ei else 1)
    if prec + rec == 0:
        F1 = 0.0
    else:
        F1 = 2 * prec * rec / (prec + rec)
    return 1.0 - F1

def compute_PDJ(src_parse, rw_parse, max_d=20):
    """Parse Depth Jump: 源句 vs 改写句深度分布差异 (L1 距离)"""
    def hist_norm(depths):
        H = [0] * (max_d + 1)
        for d in depths:
            H[min(d, max_d)] += 1
        s = sum(H) or 1
        return [h / s for h in H]
    hs, hr = hist_norm([src_parse["depth"]]), hist_norm([rw_parse["depth"]])
    return sum(abs(a - b) for a, b in zip(hs, hr))

# ---------- 3. 载入所有数据 ----------
datasets = {
    "Human":   "/root/autodl-tmp/9/data/mucgec_filtered_rewrite_path.jsonl",
    "QianWen": "/root/autodl-tmp/9/data/11qianwen_filled.jsonl",
    "WenXin":  "/root/autodl-tmp/9/data/wenxin_machine_wenxin_8.jsonl",
    "GLM":     "/root/autodl-tmp/9/data/glm4_machine_rewrites.jsonl",
    "AiYi":    "/root/autodl-tmp/9/data/rewrite_Yi.jsonl",
}

all_data = {}
total_pairs = 0
for model_name, path in datasets.items():
    data_list = load_jsonl(path)
    all_data[model_name] = data_list
    for entry in data_list:
        rewrites = entry["rewrites"]
        total_pairs += len(rewrites)

# ---------- 4. 计算并写入 CSV ----------
output_csv = "/root/autodl-tmp/9/data/syntax_metrics_per_sentence.csv"
fieldnames = [
    "Model", "source_id", "rewrite_id",
    "source", "rewrite",
    "SRM", "PathVar", "EDC", "PDJ"
]

if os.path.exists(output_csv):
    os.remove(output_csv)

with open(output_csv, "w", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

with open(output_csv, "a", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    pbar = tqdm(total=total_pairs, desc="计算句法指标进度", unit="条")
    for model_name, data_list in all_data.items():
        for src_idx, entry in enumerate(data_list):
            source_sent = entry["source"]
            rewrites = entry["rewrites"]
            if not rewrites:
                continue

            src_parse = parse_sentence(source_sent)
            rew_parse_list = [parse_sentence(r) for r in rewrites]

            path_var = compute_path_variability_weighted(rew_parse_list)

            for rw_idx, (rewrite_sent, rw_parse) in enumerate(zip(rewrites, rew_parse_list)):
                srm = compute_SRM(src_parse, rw_parse)
                edc = compute_EDC(src_parse, rw_parse)
                pdj = compute_PDJ(src_parse, rw_parse)

                row = {
                    "Model": model_name,
                    "source_id": src_idx,
                    "rewrite_id": rw_idx,
                    "source": source_sent,
                    "rewrite": rewrite_sent,
                    "SRM": srm,
                    "PathVar": path_var,
                    "EDC": edc,
                    "PDJ": pdj
                }
                writer.writerow(row)
                pbar.update(1)
    pbar.close()

print(f"✅ 已将逐句指标结果保存到：{os.path.abspath(output_csv)}")

# ---------- 5. 查看前 10 行 ----------
df = pd.read_csv(output_csv)
print("--------------")
print("已计算的模型有：", df["Model"].unique())
print("--------------")
print("\n=== 前 10 行示例 ===")
print(df[["Model","source","rewrite","SRM","PathVar","EDC","PDJ"]].head(10))
