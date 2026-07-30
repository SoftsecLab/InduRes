import os
import re
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = "/home/chy/1/data/hc3_processed/rewrites/rewrites_dev100_neutral_qwen25_7b.jsonl"
DEFAULT_OUT_DIR = "/home/chy/1/data/hc3_processed/analysis"
DEFAULT_OUT_PREFIX = "dev100_neutral_qwen25_7b_structural_v2"

DEFAULT_SPACY_MODEL = "zh_core_web_sm"
MAX_DEPTH = 10

# v2 默认边表示：
# 原版：          (head_pos, dep_label, child_pos)
# v2 默认：      (head_pos, dep_label, child_pos, direction, depth_bucket)
#
# 可选：
# pos_dep_pos
# pos_dep_pos_dir
# pos_dep_pos_depth
# pos_dep_pos_dir_depth
DEFAULT_EDGE_SCHEMA = "pos_dep_pos_dir_depth"


def load_spacy(model_name):
    import spacy
    print(f"[INFO] loading spaCy model: {model_name}")
    nlp = spacy.load(model_name)
    return nlp


def normalize_text(text):
    text = str(text)
    text = re.sub(r"\s+", "", text)
    return text


def token_is_valid(tok):
    if tok.is_space:
        return False
    if tok.is_punct:
        return False
    if str(tok.text).strip() == "":
        return False
    return True


def safe_pos(tok):
    pos = tok.pos_
    if not pos:
        pos = "X"
    return pos


def safe_dep(tok):
    dep = tok.dep_
    if not dep:
        dep = "dep"
    return dep


def compute_token_depth(tok, max_guard=100):
    """
    计算 token 在依存树中的深度。
    root 深度为 0。
    """
    depth = 0
    cur = tok
    seen = set()

    while cur.head is not cur:
        if cur.i in seen:
            break
        seen.add(cur.i)

        depth += 1
        cur = cur.head

        if depth > max_guard:
            break

    return depth


def depth_bucket(depth):
    """
    将依存深度分桶。
    这样比直接使用原始 depth 更稳，不会因为深度过细造成稀疏。
    """
    d = int(depth)

    if d <= 0:
        return "D0"
    elif d == 1:
        return "D1"
    elif d == 2:
        return "D2"
    elif d == 3:
        return "D3"
    else:
        return "D4plus"


def dependency_direction(tok):
    """
    child 相对 head 的方向。
    中文中依存方向可以反映句法组织方式。
    """
    if tok.head is tok:
        return "ROOT"

    if tok.i < tok.head.i:
        return "L"
    elif tok.i > tok.head.i:
        return "R"
    else:
        return "SELF"


def make_edge_key(tok, valid_ids, edge_schema):
    """
    构造依存边 key。

    原版：
        (head_pos, dep, child_pos)

    v2 默认：
        (head_pos, dep, child_pos, direction, depth_bucket)

    这样仍然是 POS-level dependency graph，
    但额外加入依存方向和深度层级，增强结构表达能力。
    """
    child_pos = safe_pos(tok)
    dep = safe_dep(tok)
    child_depth = compute_token_depth(tok)

    if tok.head is tok or tok.head.i not in valid_ids:
        head_pos = "ROOT"
    else:
        head_pos = safe_pos(tok.head)

    if edge_schema == "pos_dep_pos":
        return (head_pos, dep, child_pos)

    elif edge_schema == "pos_dep_pos_dir":
        direction = dependency_direction(tok)
        return (head_pos, dep, child_pos, direction)

    elif edge_schema == "pos_dep_pos_depth":
        db = depth_bucket(child_depth)
        return (head_pos, dep, child_pos, db)

    elif edge_schema == "pos_dep_pos_dir_depth":
        direction = dependency_direction(tok)
        db = depth_bucket(child_depth)
        return (head_pos, dep, child_pos, direction, db)

    else:
        raise ValueError(f"Unknown edge_schema: {edge_schema}")


def counter_total(counter_obj):
    return int(sum(counter_obj.values()))


def counter_intersection_count(c1, c2):
    total = 0
    for k in set(c1.keys()) & set(c2.keys()):
        total += min(c1[k], c2[k])
    return int(total)


def counter_l1_distance(c1, c2):
    """
    多重集合 L1 距离：
    sum_k |count1(k) - count2(k)|

    这里作为依存图边插入/删除编辑距离的近似。
    """
    total = 0
    for k in set(c1.keys()) | set(c2.keys()):
        total += abs(c1.get(k, 0) - c2.get(k, 0))
    return int(total)


def parse_dependency_graph(
    nlp,
    text,
    max_depth=MAX_DEPTH,
    edge_schema=DEFAULT_EDGE_SCHEMA
):
    """
    将文本解析为依存图近似表示。

    graph fields:
    - n_nodes: 有效 token 数
    - edges: Counter，结构依存边多重集合
    - edge_depths: dict，edge_key -> list[child_depth]
    - depth_hist: normalized depth histogram
    - raw_tokens: token text list

    v2 修改点：
    原版 edge_key 使用：
        (head_pos, dep_label, child_pos)

    v2 默认 edge_key 使用：
        (head_pos, dep_label, child_pos, direction, depth_bucket)

    这样仍然减少词面变化干扰，但比原版更能捕捉句法方向和层级结构。
    """
    doc = nlp(str(text))

    valid_ids = set()
    raw_tokens = []
    depths = []

    for tok in doc:
        if token_is_valid(tok):
            valid_ids.add(tok.i)
            raw_tokens.append(tok.text)
            depths.append(compute_token_depth(tok))

    n_nodes = len(valid_ids)

    edges = Counter()
    edge_depths = defaultdict(list)

    for tok in doc:
        if tok.i not in valid_ids:
            continue

        child_depth = compute_token_depth(tok)

        edge_key = make_edge_key(
            tok=tok,
            valid_ids=valid_ids,
            edge_schema=edge_schema
        )

        edges[edge_key] += 1
        edge_depths[edge_key].append(child_depth)

    hist = np.zeros(max_depth + 1, dtype=float)

    for d in depths:
        bucket = min(int(d), max_depth)
        hist[bucket] += 1.0

    if hist.sum() > 0:
        hist = hist / hist.sum()

    return {
        "n_nodes": n_nodes,
        "edges": edges,
        "edge_depths": dict(edge_depths),
        "depth_hist": hist,
        "raw_tokens": raw_tokens,
    }


def structural_rewriting_metric(g_source, g_rewrite):
    """
    SRM = EditDistance(G_source, G_rewrite) / Length_source

    这里将 EditDistance 实现为：
    - 依存边多重集合的 L1 距离
    - 加上节点数量差异

    这是可复现、可解释的 graph edit distance 近似。
    """
    edge_dist = counter_l1_distance(g_source["edges"], g_rewrite["edges"])
    node_dist = abs(g_source["n_nodes"] - g_rewrite["n_nodes"])

    edit_distance = edge_dist + node_dist
    length_source = max(g_source["n_nodes"], 1)

    return edit_distance / length_source


def edge_delta_core(g_source, g_rewrite):
    """
    EDC = 1 - F1

    Prec = |Ei ∩ Ej| / |Ej|
    Rec  = |Ei ∩ Ej| / |Ei|
    F1   = 2 * Prec * Rec / (Prec + Rec)
    """
    ei = g_source["edges"]
    ej = g_rewrite["edges"]

    len_i = counter_total(ei)
    len_j = counter_total(ej)

    if len_i == 0 and len_j == 0:
        return 0.0

    if len_i == 0 or len_j == 0:
        return 1.0

    overlap = counter_intersection_count(ei, ej)

    prec = overlap / len_j if len_j > 0 else 0.0
    rec = overlap / len_i if len_i > 0 else 0.0

    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    return 1.0 - f1


def dependency_depth_jaccard(g_source, g_rewrite):
    """
    PDJ = sum_d |Hs(d) - Hr(d)|

    论文里叫 Dependency Depth Jaccard，
    但公式实际是 normalized depth histogram 的 L1 distance。
    """
    hs = g_source["depth_hist"]
    hr = g_rewrite["depth_hist"]

    return float(np.sum(np.abs(hs - hr)))


def unmatched_edge_weight_sum(g1, g2):
    """
    用于 PathVar 中的 S_ij。

    Ddep(Gi, Gj) 是两个 rewrite 依存边集合的 symmetric difference。

    对每个未匹配边，按：
        1 / (1 + depth)

    加权。深层节点贡献更小，符合 PathVar 的设计。
    """
    c1 = g1["edges"]
    c2 = g2["edges"]

    d1 = g1["edge_depths"]
    d2 = g2["edge_depths"]

    score = 0.0

    all_keys = set(c1.keys()) | set(c2.keys())

    for key in all_keys:
        n1 = c1.get(key, 0)
        n2 = c2.get(key, 0)
        common = min(n1, n2)

        extra1 = n1 - common
        extra2 = n2 - common

        if extra1 > 0:
            depths = sorted(d1.get(key, []), reverse=True)
            unmatched_depths = depths[:extra1]
            for dep_depth in unmatched_depths:
                score += 1.0 / (1.0 + dep_depth)

        if extra2 > 0:
            depths = sorted(d2.get(key, []), reverse=True)
            unmatched_depths = depths[:extra2]
            for dep_depth in unmatched_depths:
                score += 1.0 / (1.0 + dep_depth)

    return score


def path_variability(rewrite_graphs):
    """
    PathVar = 1 / [n(n-1)] * sum_{i<j} S_ij * 1 / avg_nodes

    注意：
    这里严格按论文公式里的 1/[n(n-1)] 实现。
    如果按普通 pairwise average，通常会写 2/[n(n-1)]；
    但为了和论文一致，这里不用 2。
    """
    n = len(rewrite_graphs)

    if n < 2:
        return 0.0

    avg_nodes = np.mean([max(g["n_nodes"], 1) for g in rewrite_graphs])
    avg_nodes = max(float(avg_nodes), 1.0)

    total_sij = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            total_sij += unmatched_edge_weight_sum(
                rewrite_graphs[i],
                rewrite_graphs[j]
            )

    return total_sij / (n * (n - 1)) / avg_nodes


def load_rewrites_grouped(path):
    groups = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                print(f"[WARN] bad json line {line_no}: {e}")
                continue

            if item.get("status") not in {"ok", "partial_ok"}:
                continue

            if not str(item.get("rewrite_text", "")).strip():
                continue

            groups[item["task_id"]].append(item)

    for task_id in groups:
        groups[task_id] = sorted(
            groups[task_id],
            key=lambda x: x.get("rewrite_index", 0)
        )

    return groups


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default=DEFAULT_INPUT_PATH
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR
    )

    parser.add_argument(
        "--out_prefix",
        type=str,
        default=DEFAULT_OUT_PREFIX
    )

    parser.add_argument(
        "--spacy_model",
        type=str,
        default=DEFAULT_SPACY_MODEL
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=MAX_DEPTH
    )

    parser.add_argument(
        "--edge_schema",
        type=str,
        default=DEFAULT_EDGE_SCHEMA,
        choices=[
            "pos_dep_pos",
            "pos_dep_pos_dir",
            "pos_dep_pos_depth",
            "pos_dep_pos_dir_depth",
        ]
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    out_sample_csv = os.path.join(
        args.out_dir,
        f"{args.out_prefix}_sample_features.csv"
    )

    out_rewrite_csv = os.path.join(
        args.out_dir,
        f"{args.out_prefix}_rewrite_features.csv"
    )

    nlp = load_spacy(args.spacy_model)

    groups = load_rewrites_grouped(args.input_path)

    print("[INFO] input:", args.input_path)
    print("[INFO] task groups:", len(groups))
    print("[INFO] edge_schema:", args.edge_schema)
    print("[INFO] max_depth:", args.max_depth)

    graph_cache = {}

    def get_graph(text):
        key = (
            normalize_text(text),
            args.edge_schema,
            args.max_depth
        )

        if key not in graph_cache:
            graph_cache[key] = parse_dependency_graph(
                nlp=nlp,
                text=text,
                max_depth=args.max_depth,
                edge_schema=args.edge_schema
            )

        return graph_cache[key]

    sample_rows = []
    rewrite_rows = []

    for idx, (task_id, items) in enumerate(groups.items(), start=1):
        if idx % 50 == 0:
            print(f"[INFO] processed task groups: {idx}/{len(groups)}")

        first = items[0]

        source_text = first["source_text"]
        source_graph = get_graph(source_text)

        rewrite_graphs = []
        per_rewrite_metrics = []

        for item in items:
            rewrite_text = item["rewrite_text"]
            rewrite_graph = get_graph(rewrite_text)
            rewrite_graphs.append(rewrite_graph)

            srm = structural_rewriting_metric(source_graph, rewrite_graph)
            edc = edge_delta_core(source_graph, rewrite_graph)
            pdj = dependency_depth_jaccard(source_graph, rewrite_graph)

            row = {
                "rewrite_id": item.get("rewrite_id"),
                "task_id": item.get("task_id"),
                "sample_id": item.get("sample_id"),
                "topic_id": item.get("topic_id"),
                "split": item.get("split"),
                "source_label": item.get("source_label"),
                "label": item.get("label"),
                "domain": item.get("domain"),
                "prompt_type": item.get("prompt_type"),
                "rewrite_model": item.get("rewrite_model"),
                "rewrite_index": item.get("rewrite_index"),

                "edge_schema": args.edge_schema,

                "source_nodes": source_graph["n_nodes"],
                "rewrite_nodes": rewrite_graph["n_nodes"],

                "SRM": srm,
                "EDC": edc,
                "PDJ": pdj,
            }

            rewrite_rows.append(row)
            per_rewrite_metrics.append(row)

        pathvar = path_variability(rewrite_graphs)

        srm_values = [x["SRM"] for x in per_rewrite_metrics]
        edc_values = [x["EDC"] for x in per_rewrite_metrics]
        pdj_values = [x["PDJ"] for x in per_rewrite_metrics]

        sample_rows.append({
            "task_id": first.get("task_id"),
            "sample_id": first.get("sample_id"),
            "topic_id": first.get("topic_id"),
            "split": first.get("split"),
            "source_label": first.get("source_label"),
            "label": first.get("label"),
            "domain": first.get("domain"),
            "prompt_type": first.get("prompt_type"),
            "rewrite_model": first.get("rewrite_model"),

            "edge_schema": args.edge_schema,

            "n_rewrites": len(items),
            "source_nodes": source_graph["n_nodes"],
            "rewrite_nodes_mean": float(
                np.mean([g["n_nodes"] for g in rewrite_graphs])
            ),
            "rewrite_nodes_std": float(
                np.std([g["n_nodes"] for g in rewrite_graphs])
            ),

            "SRM_mean": float(np.mean(srm_values)),
            "SRM_std": float(np.std(srm_values)),
            "SRM_min": float(np.min(srm_values)),
            "SRM_max": float(np.max(srm_values)),

            "EDC_mean": float(np.mean(edc_values)),
            "EDC_std": float(np.std(edc_values)),
            "EDC_min": float(np.min(edc_values)),
            "EDC_max": float(np.max(edc_values)),

            "PDJ_mean": float(np.mean(pdj_values)),
            "PDJ_std": float(np.std(pdj_values)),
            "PDJ_min": float(np.min(pdj_values)),
            "PDJ_max": float(np.max(pdj_values)),

            "PathVar": float(pathvar),
        })

    df_sample = pd.DataFrame(sample_rows)
    df_rewrite = pd.DataFrame(rewrite_rows)

    df_sample.to_csv(
        out_sample_csv,
        index=False,
        encoding="utf-8-sig"
    )

    df_rewrite.to_csv(
        out_rewrite_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("[DONE] saved sample-level features:", out_sample_csv)
    print("[DONE] saved rewrite-level features:", out_rewrite_csv)

    print("\n[INFO] sample rows:", len(df_sample))
    print(df_sample["source_label"].value_counts())

    if "n_rewrites" in df_sample.columns:
        print("\n[INFO] n_rewrites counts:")
        print(df_sample["n_rewrites"].value_counts().sort_index())

    metric_cols = [
        "SRM_mean",
        "EDC_mean",
        "PDJ_mean",
        "PathVar",
    ]

    print("\n[INFO] group means:")
    print(df_sample.groupby("source_label")[metric_cols].mean())


if __name__ == "__main__":
    main()