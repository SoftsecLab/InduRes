#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

主要修改
--------
1. 保留论文核心指标：
   - AvgMover
   - AvgBERT
   - DC-score

2. 增加修正残差与 correction-induced drift：
   - structural_residual = PMD(O, C)
   - semantic_drift = BERT(O, P) - BERT(O, C)
   - structural_drift = PMD(O, C) - PMD(O, P)

3. 对比例指标设置最小扰动强度阈值：
   - 当 PMD(O,P) 或 Edit(O,P) 太小时，比例指标记为 NaN
   - 这些比例指标只输出观察，不进入正式候选特征组

4. 增加按扰动类型聚合的特征：
   - punctuation
   - dependency_delete
   - connective_delete

5. 正式比较固定特征组：
   - P0_paper_dc
   - P1_paper_core
   - P2_compact
   - P3_type_aware
   - P3_all_types_ablation

6. 修复交叉验证预处理泄漏：
   - 缺失值填补和标准化均放在 sklearn Pipeline 内
   - 按 topic_id 使用 StratifiedGroupKFold

7. 支持复用已有 row_metrics：
   --row_metrics_input .../dev100_row_metrics.jsonl
   这样无需重新执行 HanLP 和 BERTScore。

标签定义
--------
human source   -> label 0
machine source -> label 1

分析与训练单位
--------------
source level：每个 source 聚合其 3—4 条扰动响应。
不能把单条扰动当作相互独立的训练样本。
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EPS = 1e-12


# ============================================================
# 1. 字段候选
# ============================================================

SOURCE_FIELD_CANDIDATES = [
    "source_text",
    "original_text",
    "clean_text",
    "original",
    "source",
]

PERTURBED_FIELD_CANDIDATES = [
    "perturbed_text",
    "perturb_text",
    "corrupted_text",
    "noisy_text",
    "input_text",
    "perturbed",
]

CORRECTED_FIELD_CANDIDATES = [
    "corrected_text",
    "correction_text",
    "revised_text",
    "revision_text",
    "output_text",
    "corrected_output",
    "correction",
    "revision",
]

EDIT_RATIO_FIELD_CANDIDATES = [
    "edit_ratio",
    "correction_edit_ratio",
    "revision_edit_ratio",
]

SAMPLE_FIELD_CANDIDATES = [
    "sample_id",
    "source_id",
]

TOPIC_FIELD_CANDIDATES = [
    "topic_id",
    "pair_id",
]

LABEL_FIELD_CANDIDATES = [
    "source_label",
    "label",
]

PERTURB_TYPE_FIELD_CANDIDATES = [
    "perturbation_type",
    "perturb_type",
    "noise_type",
]


# ============================================================
# 2. 基础 I/O
# ============================================================

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(
                    f"第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc

    return rows


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_for_edit(text):
    # 只移除空白；保留标点，因为标点可能是受控扰动的一部分。
    return re.sub(r"\s+", "", str(text or ""))


def normalized_edit_ratio(text_a, text_b):
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError as exc:
        raise ImportError(
            "执行原始文本指标提取时需要 rapidfuzz："
            "pip install rapidfuzz"
        ) from exc

    text_a = normalize_for_edit(text_a)
    text_b = normalize_for_edit(text_b)

    denominator = max(
        len(text_a),
        len(text_b),
        1,
    )

    return float(
        Levenshtein.distance(text_a, text_b)
        / denominator
    )


def is_nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sanitize_feature_name(value):
    value = str(value or "unknown").strip().lower()
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def detect_text_field(
    rows,
    explicit,
    candidates,
    field_description,
):
    if explicit:
        if not any(explicit in row for row in rows):
            raise KeyError(
                f"指定的{field_description}字段不存在：{explicit}"
            )

        return explicit

    sample_rows = rows[: min(100, len(rows))]

    for candidate in candidates:
        values = [
            row.get(candidate)
            for row in sample_rows
        ]

        valid_count = sum(
            is_nonempty_string(value)
            for value in values
        )

        if valid_count >= max(
            1,
            int(0.8 * len(sample_rows)),
        ):
            return candidate

    available_keys = sorted(
        set().union(
            *(
                row.keys()
                for row in rows[: min(20, len(rows))]
            )
        )
    )

    raise KeyError(
        f"无法自动识别{field_description}字段。"
        f"当前字段包括：{available_keys}"
    )


def detect_optional_field(
    rows,
    explicit,
    candidates,
):
    if explicit:
        return explicit if any(
            explicit in row for row in rows
        ) else None

    for candidate in candidates:
        if any(candidate in row for row in rows):
            return candidate

    return None


def normalize_source_label(row, label_field=None):
    candidate_fields = []

    if label_field:
        candidate_fields.append(label_field)

    candidate_fields.extend([
        "source_label",
        "label",
    ])

    for field in dict.fromkeys(candidate_fields):
        if field not in row:
            continue

        value = row.get(field)

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in {"human", "0", "h"}:
                return "human"

            if normalized in {
                "machine",
                "1",
                "ai",
                "llm",
                "gpt",
            }:
                return "machine"

        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            return (
                "human"
                if int(value) == 0
                else "machine"
            )

    raise ValueError(
        f"无法从记录中识别 source label：{row}"
    )


# ============================================================
# 3. HanLP 依存解析
# ============================================================

def import_hanlp():
    try:
        import hanlp
    except ImportError as exc:
        raise ImportError(
            "执行原始文本指标提取时需要 HanLP："
            "pip install hanlp"
        ) from exc

    return hanlp


def default_hanlp_model(hanlp_module):
    try:
        return (
            hanlp_module.pretrained.mtl
            .CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
        )
    except Exception:
        return (
            "CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_"
            "ELECTRA_SMALL_ZH"
        )


def pick_doc_key(document, exact_keys, prefixes):
    keys = list(document.keys())

    for key in exact_keys:
        if key in document:
            return key

    for key in keys:
        key_text = str(key)

        if any(
            key_text.startswith(prefix)
            for prefix in prefixes
        ):
            return key

    return None


def is_nested_sequence(value):
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], list)
    )


def normalize_sentences(tokens, dependencies):
    if not isinstance(tokens, list):
        raise ValueError("HanLP tokens 格式异常。")

    if not isinstance(dependencies, list):
        raise ValueError("HanLP dependencies 格式异常。")

    token_sentences = (
        tokens
        if is_nested_sequence(tokens)
        else [tokens]
    )

    dependency_sentences = (
        dependencies
        if is_nested_sequence(dependencies)
        else [dependencies]
    )

    if len(token_sentences) != len(dependency_sentences):
        raise ValueError(
            "HanLP 分句数量不一致："
            f"{len(token_sentences)} vs "
            f"{len(dependency_sentences)}"
        )

    normalized_dependencies = []

    for sentence_dependencies in dependency_sentences:
        current_sentence = []

        for item in sentence_dependencies:
            if isinstance(item, dict):
                head = item.get(
                    "head",
                    item.get("HEAD", 0),
                )

                relation = (
                    item.get("deprel")
                    or item.get("rel")
                    or item.get("label")
                    or item.get("DEPREL")
                    or "dep"
                )

            elif (
                isinstance(item, (list, tuple))
                and len(item) >= 2
            ):
                head = item[0]
                relation = item[1]

            else:
                raise ValueError(
                    f"无法解析依存项：{item!r}"
                )

            current_sentence.append(
                (int(head), str(relation))
            )

        normalized_dependencies.append(
            current_sentence
        )

    return token_sentences, normalized_dependencies


def dependency_depth(
    token_index,
    heads,
    memo,
    visiting,
):
    if token_index in memo:
        return memo[token_index]

    if token_index in visiting:
        return 1

    visiting.add(token_index)
    head = heads[token_index - 1]

    if (
        head <= 0
        or head == token_index
        or head > len(heads)
    ):
        depth = 0
    else:
        depth = 1 + dependency_depth(
            head,
            heads,
            memo,
            visiting,
        )

    visiting.remove(token_index)
    memo[token_index] = depth

    return depth


def normalize_hanlp_document(document):
    if hasattr(document, "to_dict"):
        document = document.to_dict()

    if not isinstance(document, dict):
        try:
            document = dict(document)
        except Exception as exc:
            raise ValueError(
                "不支持的 HanLP 输出类型："
                f"{type(document)}"
            ) from exc

    token_key = pick_doc_key(
        document,
        exact_keys=[
            "tok/fine",
            "tok/coarse",
            "tok",
        ],
        prefixes=["tok/"],
    )

    dependency_key = pick_doc_key(
        document,
        exact_keys=["dep"],
        prefixes=["dep/"],
    )

    if token_key is None or dependency_key is None:
        raise KeyError(
            "HanLP 输出中没有找到 tok/dep。"
            f"可用键：{list(document.keys())}"
        )

    token_sentences, dependency_sentences = (
        normalize_sentences(
            document[token_key],
            document[dependency_key],
        )
    )

    nodes = []

    for sentence_tokens, sentence_dependencies in zip(
        token_sentences,
        dependency_sentences,
    ):
        if len(sentence_tokens) != len(sentence_dependencies):
            minimum_length = min(
                len(sentence_tokens),
                len(sentence_dependencies),
            )

            sentence_tokens = (
                sentence_tokens[:minimum_length]
            )

            sentence_dependencies = (
                sentence_dependencies[:minimum_length]
            )

        heads = [
            head
            for head, _ in sentence_dependencies
        ]

        relations = [
            relation
            for _, relation in sentence_dependencies
        ]

        memo = {}

        for token_index, (
            token,
            (head, relation),
        ) in enumerate(
            zip(
                sentence_tokens,
                sentence_dependencies,
            ),
            start=1,
        ):
            depth = dependency_depth(
                token_index,
                heads,
                memo,
                set(),
            )

            if head <= 0 or head > len(relations):
                parent_relation = "ROOT"
            else:
                parent_relation = relations[head - 1]

            nodes.append({
                "token": str(token),
                "dep": str(relation),
                "depth": int(depth),
                "parent_dep": str(parent_relation),
            })

    return nodes


def load_parse_cache(path):
    if not path or not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data if isinstance(data, dict) else {}


def save_parse_cache(cache, path):
    if not path:
        return

    temporary_path = f"{path}.tmp"

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cache,
            file,
            ensure_ascii=False,
        )

    os.replace(temporary_path, path)


def parse_unique_texts(
    texts,
    dependency_parser,
    cache,
    cache_path,
    save_every=100,
):
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "执行原始文本指标提取时需要 tqdm："
            "pip install tqdm"
        ) from exc

    unique_texts = list(
        dict.fromkeys(
            normalize_space(text)
            for text in texts
        )
    )

    missing_texts = [
        text
        for text in unique_texts
        if text not in cache
    ]

    print("[INFO] unique texts:", len(unique_texts))
    print(
        "[INFO] cached parses:",
        len(unique_texts) - len(missing_texts),
    )
    print(
        "[INFO] missing parses:",
        len(missing_texts),
    )

    for index, text in enumerate(
        tqdm(
            missing_texts,
            desc="HanLP parsing",
        ),
        start=1,
    ):
        try:
            document = dependency_parser(text)
            cache[text] = normalize_hanlp_document(
                document
            )
        except Exception as exc:
            print(
                "[WARN] HanLP parse failed:",
                repr(exc),
            )
            cache[text] = []

        if (
            cache_path
            and index % save_every == 0
        ):
            save_parse_cache(
                cache,
                cache_path,
            )

    save_parse_cache(
        cache,
        cache_path,
    )

    return cache


# ============================================================
# 4. Perturbation Mover Distance
# ============================================================

def structural_cost_matrix(
    nodes_a,
    nodes_b,
    dependency_weight=0.50,
    depth_weight=0.30,
    parent_weight=0.20,
):
    row_count = len(nodes_a)
    column_count = len(nodes_b)

    max_depth = max(
        1,
        max(
            [node["depth"] for node in nodes_a]
            + [0]
        ),
        max(
            [node["depth"] for node in nodes_b]
            + [0]
        ),
    )

    cost = np.zeros(
        (row_count, column_count),
        dtype=np.float64,
    )

    for row_index, node_a in enumerate(nodes_a):
        for column_index, node_b in enumerate(nodes_b):
            dependency_cost = float(
                node_a["dep"] != node_b["dep"]
            )

            depth_cost = (
                abs(
                    node_a["depth"]
                    - node_b["depth"]
                )
                / max_depth
            )

            parent_cost = float(
                node_a["parent_dep"]
                != node_b["parent_dep"]
            )

            cost[row_index, column_index] = (
                dependency_weight
                * dependency_cost
                + depth_weight
                * depth_cost
                + parent_weight
                * parent_cost
            )

    return np.clip(
        cost,
        0.0,
        1.0,
    )


def perturbation_mover_distance(
    nodes_a,
    nodes_b,
    dependency_weight=0.50,
    depth_weight=0.30,
    parent_weight=0.20,
):
    try:
        import ot
    except ImportError as exc:
        raise ImportError(
            "执行原始文本指标提取时需要 POT："
            "pip install POT"
        ) from exc

    if not nodes_a and not nodes_b:
        return 0.0

    if not nodes_a or not nodes_b:
        return 1.0

    cost = structural_cost_matrix(
        nodes_a,
        nodes_b,
        dependency_weight=dependency_weight,
        depth_weight=depth_weight,
        parent_weight=parent_weight,
    )

    source_weights = np.full(
        len(nodes_a),
        1.0 / len(nodes_a),
    )

    target_weights = np.full(
        len(nodes_b),
        1.0 / len(nodes_b),
    )

    distance = ot.emd2(
        source_weights,
        target_weights,
        cost,
    )

    return float(
        np.clip(
            distance,
            0.0,
            1.0,
        )
    )


# ============================================================
# 5. BERTScore
# ============================================================

def build_bert_scorer(
    model_type,
    num_layers,
    device,
    batch_size,
):
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise ImportError(
            "执行原始文本指标提取时需要 bert-score："
            "pip install bert-score"
        ) from exc

    return BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        lang="zh",
        rescale_with_baseline=False,
        device=device,
        batch_size=batch_size,
    )


def bertscore_f1(
    scorer,
    candidates,
    references,
):
    if len(candidates) != len(references):
        raise ValueError(
            "BERTScore candidates/references 数量不一致。"
        )

    if not candidates:
        return np.array([], dtype=float)

    _, _, f1 = scorer.score(
        candidates,
        references,
        verbose=True,
    )

    return (
        f1.detach()
        .cpu()
        .numpy()
        .astype(float)
    )


# ============================================================
# 6. row-level 指标
# ============================================================

def thresholded_ratio(
    numerator,
    denominator,
    minimum_denominator,
):
    if not np.isfinite(numerator):
        return np.nan

    if not np.isfinite(denominator):
        return np.nan

    if denominator < minimum_denominator:
        return np.nan

    return float(
        numerator / denominator
    )


def enrich_row_metric(
    row,
    min_struct_damage,
    min_char_damage,
):
    """
    对新生成或旧版 row_metrics 统一重算派生指标。
    """
    result = dict(row)

    required = [
        "pmd_pc",
        "pmd_op",
        "pmd_oc",
        "bert_pc",
        "bert_op",
        "bert_oc",
        "edit_ratio_pc",
        "edit_ratio_op",
        "edit_ratio_oc",
    ]

    missing = [
        field
        for field in required
        if field not in result
    ]

    if missing:
        raise KeyError(
            "row_metrics 缺少必要字段："
            f"{missing}"
        )

    for field in required:
        result[field] = safe_float(
            result[field],
            default=np.nan,
        )

    result["source_label"] = normalize_source_label(
        result
    )

    result["label"] = (
        0
        if result["source_label"] == "human"
        else 1
    )

    result["topic_id"] = str(
        result.get("topic_id", "")
    )

    result["sample_id"] = str(
        result.get(
            "sample_id",
            (
                f"{result['topic_id']}_"
                f"{result['source_label']}"
            ),
        )
    )

    result["perturbation_type"] = sanitize_feature_name(
        result.get(
            "perturbation_type",
            "unknown",
        )
    )

    # P -> C 的修正行为
    result["mover_score"] = float(
        1.0 - result["pmd_pc"]
    )

    result["corrected_flag"] = int(
        result["edit_ratio_pc"] > EPS
    )

    # recovery：正值表示修正后更接近原始文本
    result["structural_recovery"] = float(
        result["pmd_op"]
        - result["pmd_oc"]
    )

    result["semantic_recovery"] = float(
        result["bert_oc"]
        - result["bert_op"]
    )

    result["char_recovery"] = float(
        result["edit_ratio_op"]
        - result["edit_ratio_oc"]
    )

    # drift：正值表示修正后反而离原始文本更远
    result["structural_drift"] = float(
        result["pmd_oc"]
        - result["pmd_op"]
    )

    result["semantic_drift"] = float(
        result["bert_op"]
        - result["bert_oc"]
    )

    result["char_drift"] = float(
        result["edit_ratio_oc"]
        - result["edit_ratio_op"]
    )

    # 比例指标仅在实际扰动强度达到阈值时计算
    result["structural_recovery_rate"] = (
        thresholded_ratio(
            result["structural_recovery"],
            result["pmd_op"],
            min_struct_damage,
        )
    )

    result["structural_drift_rate"] = (
        thresholded_ratio(
            result["structural_drift"],
            result["pmd_op"],
            min_struct_damage,
        )
    )

    result["char_recovery_rate"] = (
        thresholded_ratio(
            result["char_recovery"],
            result["edit_ratio_op"],
            min_char_damage,
        )
    )

    result["char_drift_rate"] = (
        thresholded_ratio(
            result["char_drift"],
            result["edit_ratio_op"],
            min_char_damage,
        )
    )

    result["relative_correction"] = (
        thresholded_ratio(
            result["edit_ratio_pc"],
            result["edit_ratio_op"],
            min_char_damage,
        )
    )

    return result


def build_row_metrics_from_raw(
    raw_rows,
    fields,
    parse_cache,
    bert_scorer,
    args,
):
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "执行原始文本指标提取时需要 tqdm："
            "pip install tqdm"
        ) from exc

    originals = [
        normalize_space(
            row[fields["source"]]
        )
        for row in raw_rows
    ]

    perturbed_texts = [
        normalize_space(
            row[fields["perturbed"]]
        )
        for row in raw_rows
    ]

    corrected_texts = [
        normalize_space(
            row[fields["corrected"]]
        )
        for row in raw_rows
    ]

    print(
        "[INFO] BERTScore: perturbed vs corrected"
    )

    bert_pc = bertscore_f1(
        bert_scorer,
        candidates=corrected_texts,
        references=perturbed_texts,
    )

    print(
        "[INFO] BERTScore: original vs perturbed"
    )

    bert_op = bertscore_f1(
        bert_scorer,
        candidates=perturbed_texts,
        references=originals,
    )

    print(
        "[INFO] BERTScore: original vs corrected"
    )

    bert_oc = bertscore_f1(
        bert_scorer,
        candidates=corrected_texts,
        references=originals,
    )

    output_rows = []

    for index, raw_row in enumerate(
        tqdm(
            raw_rows,
            desc="Pair metrics",
        )
    ):
        original = originals[index]
        perturbed = perturbed_texts[index]
        corrected = corrected_texts[index]

        original_nodes = parse_cache.get(
            original,
            [],
        )

        perturbed_nodes = parse_cache.get(
            perturbed,
            [],
        )

        corrected_nodes = parse_cache.get(
            corrected,
            [],
        )

        pmd_pc = perturbation_mover_distance(
            perturbed_nodes,
            corrected_nodes,
            dependency_weight=(
                args.dependency_weight
            ),
            depth_weight=args.depth_weight,
            parent_weight=args.parent_weight,
        )

        pmd_op = perturbation_mover_distance(
            original_nodes,
            perturbed_nodes,
            dependency_weight=(
                args.dependency_weight
            ),
            depth_weight=args.depth_weight,
            parent_weight=args.parent_weight,
        )

        pmd_oc = perturbation_mover_distance(
            original_nodes,
            corrected_nodes,
            dependency_weight=(
                args.dependency_weight
            ),
            depth_weight=args.depth_weight,
            parent_weight=args.parent_weight,
        )

        existing_edit_ratio = np.nan

        if fields["edit_ratio"]:
            existing_edit_ratio = safe_float(
                raw_row.get(
                    fields["edit_ratio"]
                ),
                default=np.nan,
            )

        edit_ratio_pc = (
            existing_edit_ratio
            if np.isfinite(existing_edit_ratio)
            else normalized_edit_ratio(
                perturbed,
                corrected,
            )
        )

        edit_ratio_op = normalized_edit_ratio(
            original,
            perturbed,
        )

        edit_ratio_oc = normalized_edit_ratio(
            original,
            corrected,
        )

        source_label = normalize_source_label(
            raw_row,
            fields["label"],
        )

        topic_id = str(
            raw_row.get(
                fields["topic"],
                "",
            )
        )

        if fields["sample"]:
            sample_id = str(
                raw_row.get(
                    fields["sample"],
                    "",
                )
            )
        else:
            sample_id = (
                f"{topic_id}_{source_label}"
            )

        perturbation_type = (
            str(
                raw_row.get(
                    fields["perturb_type"],
                    "unknown",
                )
            )
            if fields["perturb_type"]
            else "unknown"
        )

        base_row = {
            "row_index": index,
            "topic_id": topic_id,
            "sample_id": sample_id,
            "source_label": source_label,
            "label": (
                0
                if source_label == "human"
                else 1
            ),
            "perturbation_type": perturbation_type,

            "edit_ratio_pc": float(
                edit_ratio_pc
            ),
            "pmd_pc": float(pmd_pc),
            "bert_pc": float(
                bert_pc[index]
            ),

            "pmd_op": float(pmd_op),
            "bert_op": float(
                bert_op[index]
            ),
            "edit_ratio_op": float(
                edit_ratio_op
            ),

            "pmd_oc": float(pmd_oc),
            "bert_oc": float(
                bert_oc[index]
            ),
            "edit_ratio_oc": float(
                edit_ratio_oc
            ),
        }

        output_rows.append(
            enrich_row_metric(
                base_row,
                min_struct_damage=(
                    args.min_struct_damage
                ),
                min_char_damage=(
                    args.min_char_damage
                ),
            )
        )

    return output_rows


def load_and_enrich_existing_row_metrics(
    path,
    args,
):
    existing_rows = load_jsonl(path)

    enriched_rows = [
        enrich_row_metric(
            row,
            min_struct_damage=(
                args.min_struct_damage
            ),
            min_char_damage=(
                args.min_char_damage
            ),
        )
        for row in existing_rows
    ]

    return enriched_rows


# ============================================================
# 7. source-level 聚合
# ============================================================

def finite_values(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return values[
        np.isfinite(values)
    ]


def summarize(
    values,
    prefix,
    output,
):
    values = finite_values(values)

    output[
        f"{prefix}_valid_n"
    ] = int(len(values))

    if len(values) == 0:
        for suffix in [
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]:
            output[
                f"{prefix}_{suffix}"
            ] = np.nan

        return

    output[f"{prefix}_mean"] = float(
        np.mean(values)
    )

    output[f"{prefix}_median"] = float(
        np.median(values)
    )

    output[f"{prefix}_std"] = (
        float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else 0.0
    )

    output[f"{prefix}_min"] = float(
        np.min(values)
    )

    output[f"{prefix}_max"] = float(
        np.max(values)
    )


GLOBAL_METRICS = {
    "edit_ratio": "edit_ratio_pc",
    "mover": "mover_score",
    "bert": "bert_pc",

    "perturbation_damage_struct": "pmd_op",
    "structural_residual": "pmd_oc",

    "structural_recovery": "structural_recovery",
    "structural_drift": "structural_drift",

    "structural_recovery_rate": (
        "structural_recovery_rate"
    ),
    "structural_drift_rate": (
        "structural_drift_rate"
    ),

    "perturbation_semantic_fidelity": "bert_op",
    "corrected_semantic_fidelity": "bert_oc",

    "semantic_recovery": "semantic_recovery",
    "semantic_drift": "semantic_drift",

    "perturbation_damage_char": "edit_ratio_op",
    "char_residual": "edit_ratio_oc",

    "char_recovery": "char_recovery",
    "char_drift": "char_drift",

    "char_recovery_rate": "char_recovery_rate",
    "char_drift_rate": "char_drift_rate",

    "relative_correction": "relative_correction",
}


TYPE_SPECIFIC_METRICS = {
    "edit_ratio": "edit_ratio_pc",
    "mover": "mover_score",
    "bert": "bert_pc",
    "structural_residual": "pmd_oc",
    "corrected_semantic_fidelity": "bert_oc",
    "structural_drift": "structural_drift",
    "semantic_drift": "semantic_drift",
}


def aggregate_source_features(
    row_metrics,
    lambda_dc,
):
    groups = defaultdict(list)

    for row in row_metrics:
        key = (
            row["topic_id"],
            row["sample_id"],
            row["source_label"],
            row["label"],
        )

        groups[key].append(row)

    source_rows = []

    for (
        topic_id,
        sample_id,
        source_label,
        label,
    ), items in groups.items():
        feature_row = {
            "topic_id": topic_id,
            "sample_id": sample_id,
            "source_label": source_label,
            "label": label,
            "n_perturbations": len(items),
        }

        for prefix, row_field in GLOBAL_METRICS.items():
            summarize(
                [
                    item.get(
                        row_field,
                        np.nan,
                    )
                    for item in items
                ],
                prefix,
                feature_row,
            )

        edit_values = finite_values([
            item.get(
                "edit_ratio_pc",
                np.nan,
            )
            for item in items
        ])

        nonzero_edit_values = edit_values[
            edit_values > EPS
        ]

        feature_row[
            "edit_ratio_nonzero_mean"
        ] = (
            float(
                np.mean(
                    nonzero_edit_values
                )
            )
            if len(nonzero_edit_values) > 0
            else 0.0
        )

        feature_row[
            "edit_ratio_nonzero_n"
        ] = int(
            len(nonzero_edit_values)
        )

        feature_row[
            "correction_rate"
        ] = float(
            np.mean([
                item.get(
                    "corrected_flag",
                    0,
                )
                for item in items
            ])
        )

        # 论文核心指标
        feature_row["avg_mover"] = (
            feature_row["mover_mean"]
        )

        feature_row["avg_bert"] = (
            feature_row["bert_mean"]
        )

        feature_row["dc_score"] = float(
            lambda_dc
            * feature_row["avg_mover"]
            + (1.0 - lambda_dc)
            * feature_row["avg_bert"]
        )

        # 按扰动类型聚合
        type_groups = defaultdict(list)

        for item in items:
            perturbation_type = (
                sanitize_feature_name(
                    item.get(
                        "perturbation_type",
                        "unknown",
                    )
                )
            )

            type_groups[
                perturbation_type
            ].append(item)

        for (
            perturbation_type,
            type_items,
        ) in type_groups.items():
            type_prefix = (
                f"ptype_{perturbation_type}"
            )

            feature_row[
                f"{type_prefix}_n"
            ] = len(type_items)

            for (
                metric_name,
                row_field,
            ) in TYPE_SPECIFIC_METRICS.items():
                summarize(
                    [
                        item.get(
                            row_field,
                            np.nan,
                        )
                        for item in type_items
                    ],
                    (
                        f"{type_prefix}_"
                        f"{metric_name}"
                    ),
                    feature_row,
                )

        source_rows.append(feature_row)

    dataframe = pd.DataFrame(source_rows)

    # 保证所有 source 都具有相同的列集合。
    dataframe = dataframe.sort_values(
        [
            "topic_id",
            "source_label",
        ]
    ).reset_index(drop=True)

    return dataframe


# ============================================================
# 8. 配对单指标区分度
# ============================================================

META_COLUMNS = {
    "topic_id",
    "sample_id",
    "source_label",
    "label",
    "n_perturbations",
}


# 这些比例指标仍保存在 source_features 中供诊断，
# 但由于分母接近 0 时不稳定，不作为正式候选指标排序或训练。
UNSTABLE_RATIO_PREFIXES = (
    "structural_recovery_rate",
    "structural_drift_rate",
    "char_recovery_rate",
    "char_drift_rate",
    "relative_correction",
)


def feature_eligibility(feature):
    if feature.startswith(UNSTABLE_RATIO_PREFIXES):
        return False, "ratio feature excluded from formal training"

    if feature == "correction_rate":
        return False, "binary correction occurrence retained for ablation only"

    return True, ""


def is_candidate_numeric_column(
    dataframe,
    column,
):
    if column in META_COLUMNS:
        return False

    if column.endswith("_valid_n"):
        return False

    if column.endswith("_n"):
        return False

    return pd.api.types.is_numeric_dtype(
        dataframe[column]
    )


def numeric_feature_columns(dataframe):
    return [
        column
        for column in dataframe.columns
        if is_candidate_numeric_column(
            dataframe,
            column,
        )
    ]


def paired_feature_arrays(
    dataframe,
    feature,
):
    pivot = dataframe.pivot_table(
        index="topic_id",
        columns="source_label",
        values=feature,
        aggfunc="first",
    )

    if (
        "human" not in pivot.columns
        or "machine" not in pivot.columns
    ):
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    pivot = pivot.dropna(
        subset=[
            "human",
            "machine",
        ]
    )

    return (
        pivot["human"].to_numpy(
            dtype=float
        ),
        pivot["machine"].to_numpy(
            dtype=float
        ),
    )


def safe_wilcoxon(
    human_values,
    machine_values,
):
    mask = (
        np.isfinite(human_values)
        & np.isfinite(machine_values)
    )

    human_values = human_values[mask]
    machine_values = machine_values[mask]

    if len(human_values) == 0:
        return np.nan, 1.0

    if np.allclose(
        human_values,
        machine_values,
    ):
        return np.nan, 1.0

    try:
        statistic, p_value = wilcoxon(
            human_values,
            machine_values,
            zero_method="wilcox",
            alternative="two-sided",
        )

        return (
            float(statistic),
            float(p_value),
        )

    except ValueError:
        return np.nan, 1.0


def single_feature_report(
    dataframe,
    split_role,
):
    report_rows = []

    for feature in numeric_feature_columns(
        dataframe
    ):
        human_values, machine_values = (
            paired_feature_arrays(
                dataframe,
                feature,
            )
        )

        mask = (
            np.isfinite(human_values)
            & np.isfinite(machine_values)
        )

        human_values = human_values[mask]
        machine_values = machine_values[mask]

        if len(human_values) == 0:
            continue

        labels = np.concatenate([
            np.zeros(
                len(human_values),
                dtype=int,
            ),
            np.ones(
                len(machine_values),
                dtype=int,
            ),
        ])

        scores = np.concatenate([
            human_values,
            machine_values,
        ])

        if np.allclose(
            scores,
            scores[0],
        ):
            auc_higher_predicts_machine = 0.5
        else:
            auc_higher_predicts_machine = float(
                roc_auc_score(
                    labels,
                    scores,
                )
            )

        auc_lower_predicts_machine = float(
            1.0
            - auc_higher_predicts_machine
        )

        statistic, p_value = safe_wilcoxon(
            human_values,
            machine_values,
        )

        paired_difference = (
            human_values
            - machine_values
        )

        eligible, exclusion_reason = feature_eligibility(
            feature
        )

        row = {
            "feature": feature,
            "eligible_for_training": eligible,
            "exclusion_reason": exclusion_reason,
            "n_paired_topics": len(
                human_values
            ),

            "human_mean": float(
                np.mean(human_values)
            ),
            "human_median": float(
                np.median(human_values)
            ),
            "human_std": (
                float(
                    np.std(
                        human_values,
                        ddof=1,
                    )
                )
                if len(human_values) > 1
                else 0.0
            ),

            "machine_mean": float(
                np.mean(machine_values)
            ),
            "machine_median": float(
                np.median(machine_values)
            ),
            "machine_std": (
                float(
                    np.std(
                        machine_values,
                        ddof=1,
                    )
                )
                if len(machine_values) > 1
                else 0.0
            ),

            "paired_mean_human_minus_machine": float(
                np.mean(
                    paired_difference
                )
            ),

            "paired_median_human_minus_machine": float(
                np.median(
                    paired_difference
                )
            ),

            "human_gt_machine_rate": float(
                np.mean(
                    paired_difference > EPS
                )
            ),

            "equal_rate": float(
                np.mean(
                    np.isclose(
                        paired_difference,
                        0.0,
                    )
                )
            ),

            "human_lt_machine_rate": float(
                np.mean(
                    paired_difference < -EPS
                )
            ),

            "wilcoxon_statistic": statistic,
            "wilcoxon_p": p_value,

            # label 1 = machine
            "auc_higher_predicts_machine": (
                auc_higher_predicts_machine
            ),
            "auc_lower_predicts_machine": (
                auc_lower_predicts_machine
            ),
        }

        if split_role in {"train", "dev"}:
            if (
                auc_higher_predicts_machine
                >= auc_lower_predicts_machine
            ):
                row[
                    "best_direction_exploratory"
                ] = "higher predicts machine"

                row[
                    "directional_auc_exploratory"
                ] = auc_higher_predicts_machine
            else:
                row[
                    "best_direction_exploratory"
                ] = "lower predicts machine"

                row[
                    "directional_auc_exploratory"
                ] = auc_lower_predicts_machine

        report_rows.append(row)

    report = pd.DataFrame(report_rows)

    if report.empty:
        return report

    if split_role in {"train", "dev"}:
        report = report.sort_values(
            [
                "eligible_for_training",
                "directional_auc_exploratory",
                "wilcoxon_p",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
    else:
        # test 不重新选择方向，仅按固定的 raw AUC 展示。
        report = report.sort_values(
            "auc_higher_predicts_machine",
            ascending=False,
        )

    return report.reset_index(drop=True)


# ============================================================
# 9. DC-score lambda 扫描
# ============================================================

def dc_lambda_scan(dataframe):
    human = (
        dataframe[
            dataframe["source_label"]
            == "human"
        ]
        .set_index("topic_id")
    )

    machine = (
        dataframe[
            dataframe["source_label"]
            == "machine"
        ]
        .set_index("topic_id")
    )

    common_topics = (
        human.index.intersection(
            machine.index
        )
    )

    human = human.loc[common_topics]
    machine = machine.loc[common_topics]

    labels = np.concatenate([
        np.zeros(
            len(human),
            dtype=int,
        ),
        np.ones(
            len(machine),
            dtype=int,
        ),
    ])

    rows = []

    for lambda_dc in np.linspace(
        0.0,
        1.0,
        21,
    ):
        human_dc = (
            lambda_dc
            * human["avg_mover"].to_numpy()
            + (1.0 - lambda_dc)
            * human["avg_bert"].to_numpy()
        )

        machine_dc = (
            lambda_dc
            * machine["avg_mover"].to_numpy()
            + (1.0 - lambda_dc)
            * machine["avg_bert"].to_numpy()
        )

        scores = np.concatenate([
            human_dc,
            machine_dc,
        ])

        if np.allclose(
            scores,
            scores[0],
        ):
            auc = 0.5
        else:
            auc = float(
                roc_auc_score(
                    labels,
                    scores,
                )
            )

        rows.append({
            "lambda_dc": float(
                lambda_dc
            ),
            "auc_higher_dc_predicts_machine": (
                auc
            ),
            "auc_lower_dc_predicts_machine": (
                1.0 - auc
            ),
            "directional_auc_exploratory": max(
                auc,
                1.0 - auc,
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "directional_auc_exploratory",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# 10. 固定特征组
# ============================================================

P0_PAPER_DC = [
    "dc_score",
]

P1_PAPER_CORE = [
    "avg_mover",
    "avg_bert",
]

P2_COMPACT = [
    "avg_mover",
    "avg_bert",

    "structural_residual_mean",
    "corrected_semantic_fidelity_std",

    "edit_ratio_std",
    "edit_ratio_nonzero_mean",

    # 使用 drift 正方向表达“过度修正”
    "semantic_drift_mean",
    "structural_drift_mean",
]

P3_TYPE_AWARE_EXTRA = [
    "ptype_punctuation_structural_residual_mean",
    "ptype_dependency_delete_structural_residual_mean",

    "ptype_punctuation_corrected_semantic_fidelity_mean",
    "ptype_dependency_delete_corrected_semantic_fidelity_mean",

    "ptype_punctuation_mover_mean",
    "ptype_dependency_delete_mover_mean",

    "ptype_punctuation_edit_ratio_mean",
    "ptype_dependency_delete_edit_ratio_mean",
]

P3_CONNECTIVE_ABLATION_EXTRA = [
    "ptype_connective_delete_structural_residual_mean",
    "ptype_connective_delete_corrected_semantic_fidelity_mean",
    "ptype_connective_delete_mover_mean",
    "ptype_connective_delete_edit_ratio_mean",
]

FEATURE_SETS = {
    "edit_single_baseline": [
        "edit_ratio_mean",
    ],

    "P0_paper_dc": P0_PAPER_DC,

    "P1_paper_core": P1_PAPER_CORE,

    "P2_compact": P2_COMPACT,

    "P3_type_aware": (
        P2_COMPACT
        + P3_TYPE_AWARE_EXTRA
    ),

    # 用于验证 connective_delete 是否真正有增益
    "P3_all_types_ablation": (
        P2_COMPACT
        + P3_TYPE_AWARE_EXTRA
        + P3_CONNECTIVE_ABLATION_EXTRA
    ),
}


def available_feature_set(
    dataframe,
    requested_features,
):
    available = [
        feature
        for feature in requested_features
        if (
            feature in dataframe.columns
            and pd.api.types.is_numeric_dtype(
                dataframe[feature]
            )
        )
    ]

    missing = [
        feature
        for feature in requested_features
        if feature not in available
    ]

    return available, missing


# ============================================================
# 11. Grouped CV
# ============================================================

def build_models(random_state):
    return {
        "logistic": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]),

        "rbf_svm": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=False,
                    random_state=random_state,
                ),
            ),
        ]),
    }


def grouped_cv_report(
    dataframe,
    random_state=42,
    n_splits=5,
):
    labels = dataframe[
        "label"
    ].to_numpy(dtype=int)

    groups = dataframe[
        "topic_id"
    ].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    models = build_models(
        random_state
    )

    report_rows = []

    for (
        feature_set_name,
        requested_features,
    ) in FEATURE_SETS.items():
        features, missing_features = (
            available_feature_set(
                dataframe,
                requested_features,
            )
        )

        if missing_features:
            print(
                f"[WARN] {feature_set_name} "
                f"缺少特征：{missing_features}"
            )

        if not features:
            continue

        feature_matrix = (
            dataframe[features]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .to_numpy(dtype=float)
        )

        for (
            model_name,
            model_template,
        ) in models.items():
            fold_aucs = []
            fold_accuracies = []
            fold_weighted_f1 = []

            for (
                train_index,
                validation_index,
            ) in splitter.split(
                feature_matrix,
                labels,
                groups,
            ):
                model = clone(
                    model_template
                )

                model.fit(
                    feature_matrix[
                        train_index
                    ],
                    labels[train_index],
                )

                scores = model.decision_function(
                    feature_matrix[
                        validation_index
                    ]
                )

                predictions = (
                    scores >= 0.0
                ).astype(int)

                fold_aucs.append(
                    roc_auc_score(
                        labels[
                            validation_index
                        ],
                        scores,
                    )
                )

                fold_accuracies.append(
                    accuracy_score(
                        labels[
                            validation_index
                        ],
                        predictions,
                    )
                )

                fold_weighted_f1.append(
                    f1_score(
                        labels[
                            validation_index
                        ],
                        predictions,
                        average="weighted",
                    )
                )

            report_rows.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "n_requested_features": len(
                    requested_features
                ),
                "n_used_features": len(
                    features
                ),
                "missing_features": ",".join(
                    missing_features
                ),
                "features": ",".join(
                    features
                ),

                "cv_auc_mean": float(
                    np.mean(
                        fold_aucs
                    )
                ),
                "cv_auc_std": float(
                    np.std(
                        fold_aucs,
                        ddof=1,
                    )
                ),

                "cv_accuracy_mean": float(
                    np.mean(
                        fold_accuracies
                    )
                ),
                "cv_accuracy_std": float(
                    np.std(
                        fold_accuracies,
                        ddof=1,
                    )
                ),

                "cv_weighted_f1_mean": float(
                    np.mean(
                        fold_weighted_f1
                    )
                ),
                "cv_weighted_f1_std": float(
                    np.std(
                        fold_weighted_f1,
                        ddof=1,
                    )
                ),
            })

    report = pd.DataFrame(report_rows)

    if report.empty:
        return report

    return (
        report.sort_values(
            [
                "cv_auc_mean",
                "cv_weighted_f1_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================
# 12. 特征相关性
# ============================================================

def candidate_union(dataframe):
    requested = []

    for features in FEATURE_SETS.values():
        requested.extend(features)

    requested = list(
        dict.fromkeys(requested)
    )

    return [
        feature
        for feature in requested
        if feature in dataframe.columns
    ]


def feature_correlation_reports(
    dataframe,
    threshold=0.90,
):
    features = candidate_union(
        dataframe
    )

    if len(features) < 2:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    correlation = (
        dataframe[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .corr(
            method="spearman",
            min_periods=20,
        )
    )

    high_pairs = []

    for row_index, feature_a in enumerate(features):
        for feature_b in features[
            row_index + 1:
        ]:
            rho = correlation.loc[
                feature_a,
                feature_b,
            ]

            if (
                np.isfinite(rho)
                and abs(rho) >= threshold
            ):
                high_pairs.append({
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "spearman_rho": float(rho),
                    "abs_spearman_rho": float(
                        abs(rho)
                    ),
                })

    high_pairs_dataframe = pd.DataFrame(
        high_pairs
    )

    if not high_pairs_dataframe.empty:
        high_pairs_dataframe = (
            high_pairs_dataframe.sort_values(
                "abs_spearman_rho",
                ascending=False,
            )
        )

    return (
        correlation,
        high_pairs_dataframe,
    )


# ============================================================
# 13. 检查配对设计
# ============================================================

def check_pairing(source_dataframe):
    label_counts = (
        source_dataframe[
            "source_label"
        ]
        .value_counts()
        .to_dict()
    )

    topic_label_counts = (
        source_dataframe
        .groupby(
            [
                "topic_id",
                "source_label",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    valid_pair_mask = (
        (topic_label_counts.get("human", 0) == 1)
        & (
            topic_label_counts.get(
                "machine",
                0,
            )
            == 1
        )
    )

    valid_topics = (
        topic_label_counts.index[
            valid_pair_mask
        ]
    )

    valid_dataframe = (
        source_dataframe[
            source_dataframe[
                "topic_id"
            ].isin(valid_topics)
        ]
        .copy()
    )

    print(
        "[INFO] source label counts:",
        label_counts,
    )

    print(
        "[INFO] valid paired topics:",
        len(valid_topics),
    )

    if len(valid_dataframe) != len(source_dataframe):
        print(
            "[WARN] 删除非严格配对的 source rows：",
            (
                len(source_dataframe)
                - len(valid_dataframe)
            ),
        )

    return valid_dataframe


# ============================================================
# 14. 主程序
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        default=None,
        help=(
            "原始 correction JSONL。"
            "未指定 --row_metrics_input 时必填。"
        ),
    )

    parser.add_argument(
        "--row_metrics_input",
        default=None,
        help=(
            "已有 *_row_metrics.jsonl。"
            "指定后跳过 HanLP 和 BERTScore。"
        ),
    )

    parser.add_argument(
        "--out_prefix",
        required=True,
    )

    parser.add_argument(
        "--split_role",
        choices=[
            "train",
            "dev",
            "test",
        ],
        default="dev",
    )

    parser.add_argument(
        "--source_field",
        default=None,
    )

    parser.add_argument(
        "--perturbed_field",
        default=None,
    )

    parser.add_argument(
        "--corrected_field",
        default=None,
    )

    parser.add_argument(
        "--edit_ratio_field",
        default=None,
    )

    parser.add_argument(
        "--sample_field",
        default=None,
    )

    parser.add_argument(
        "--topic_field",
        default=None,
    )

    parser.add_argument(
        "--label_field",
        default=None,
    )

    parser.add_argument(
        "--perturb_type_field",
        default=None,
    )

    parser.add_argument(
        "--bert_model",
        default="bert-base-chinese",
    )

    parser.add_argument(
        "--bert_num_layers",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--bert_batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--hanlp_model",
        default=None,
    )

    parser.add_argument(
        "--parse_cache",
        default=None,
    )

    parser.add_argument(
        "--lambda_dc",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--dependency_weight",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--depth_weight",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--parent_weight",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--min_struct_damage",
        type=float,
        default=0.01,
        help=(
            "PMD(O,P) 小于该值时，"
            "结构比例指标记为 NaN。"
        ),
    )

    parser.add_argument(
        "--min_char_damage",
        type=float,
        default=0.01,
        help=(
            "Edit(O,P) 小于该值时，"
            "字符比例指标记为 NaN。"
        ),
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--cv_splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--correlation_threshold",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--skip_cv",
        action="store_true",
    )

    return parser.parse_args()


def validate_arguments(args):
    if (
        args.row_metrics_input is None
        and args.input_path is None
    ):
        raise ValueError(
            "--input_path 与 --row_metrics_input "
            "至少指定一个。"
        )

    if not 0.0 <= args.lambda_dc <= 1.0:
        raise ValueError(
            "--lambda_dc 必须在 [0,1] 内。"
        )

    weight_sum = (
        args.dependency_weight
        + args.depth_weight
        + args.parent_weight
    )

    if not np.isclose(
        weight_sum,
        1.0,
    ):
        raise ValueError(
            "dependency/depth/parent 三个权重"
            "必须和为 1。"
            f"当前为 {weight_sum}"
        )

    if args.min_struct_damage < 0:
        raise ValueError(
            "--min_struct_damage 不能为负。"
        )

    if args.min_char_damage < 0:
        raise ValueError(
            "--min_char_damage 不能为负。"
        )


def extract_row_metrics(args):
    raw_rows = load_jsonl(
        args.input_path
    )

    if not raw_rows:
        raise ValueError(
            "原始 correction JSONL 为空。"
        )

    fields = {
        "source": detect_text_field(
            raw_rows,
            args.source_field,
            SOURCE_FIELD_CANDIDATES,
            "原始文本",
        ),

        "perturbed": detect_text_field(
            raw_rows,
            args.perturbed_field,
            PERTURBED_FIELD_CANDIDATES,
            "扰动文本",
        ),

        "corrected": detect_text_field(
            raw_rows,
            args.corrected_field,
            CORRECTED_FIELD_CANDIDATES,
            "修正文本",
        ),

        "edit_ratio": detect_optional_field(
            raw_rows,
            args.edit_ratio_field,
            EDIT_RATIO_FIELD_CANDIDATES,
        ),

        "sample": detect_optional_field(
            raw_rows,
            args.sample_field,
            SAMPLE_FIELD_CANDIDATES,
        ),

        "topic": detect_optional_field(
            raw_rows,
            args.topic_field,
            TOPIC_FIELD_CANDIDATES,
        ),

        "label": detect_optional_field(
            raw_rows,
            args.label_field,
            LABEL_FIELD_CANDIDATES,
        ),

        "perturb_type": detect_optional_field(
            raw_rows,
            args.perturb_type_field,
            PERTURB_TYPE_FIELD_CANDIDATES,
        ),
    }

    if fields["topic"] is None:
        raise KeyError(
            "未识别 topic_id；"
            "请用 --topic_field 指定。"
        )

    if fields["label"] is None:
        raise KeyError(
            "未识别 source_label/label；"
            "请用 --label_field 指定。"
        )

    print("[INFO] detected fields:")

    for key, value in fields.items():
        print(f"  {key}: {value}")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "执行原始文本指标提取时需要 torch。"
        ) from exc

    if args.device is None:
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    hanlp_module = import_hanlp()

    hanlp_model = (
        args.hanlp_model
        if args.hanlp_model
        else default_hanlp_model(
            hanlp_module
        )
    )

    print(
        "[INFO] loading HanLP:",
        hanlp_model,
    )

    dependency_parser = hanlp_module.load(
        hanlp_model
    )

    parse_cache_path = (
        args.parse_cache
        if args.parse_cache
        else (
            f"{args.out_prefix}"
            "_hanlp_cache.json"
        )
    )

    parse_cache = load_parse_cache(
        parse_cache_path
    )

    all_texts = []

    for row in raw_rows:
        all_texts.extend([
            row[fields["source"]],
            row[fields["perturbed"]],
            row[fields["corrected"]],
        ])

    parse_cache = parse_unique_texts(
        all_texts,
        dependency_parser,
        parse_cache,
        parse_cache_path,
    )

    print(
        "[INFO] loading BERTScore:",
        args.bert_model,
    )

    bert_scorer = build_bert_scorer(
        model_type=args.bert_model,
        num_layers=args.bert_num_layers,
        device=args.device,
        batch_size=args.bert_batch_size,
    )

    return build_row_metrics_from_raw(
        raw_rows,
        fields,
        parse_cache,
        bert_scorer,
        args,
    )


def print_top_single_features(
    report,
    split_role,
):
    if report.empty:
        print("[WARN] 单指标报告为空。")
        return

    if split_role in {"train", "dev"}:
        display_columns = [
            "feature",
            "eligible_for_training",
            "n_paired_topics",
            "human_mean",
            "machine_mean",
            "wilcoxon_p",
            "auc_higher_predicts_machine",
            "auc_lower_predicts_machine",
            "directional_auc_exploratory",
            "best_direction_exploratory",
        ]
    else:
        display_columns = [
            "feature",
            "n_paired_topics",
            "human_mean",
            "machine_mean",
            "wilcoxon_p",
            "auc_higher_predicts_machine",
        ]

    print()
    print("=" * 100)
    print("Top single features")
    print("=" * 100)

    display_report = report

    if "eligible_for_training" in display_report.columns:
        display_report = display_report[
            display_report["eligible_for_training"]
        ]

    print(
        display_report[
            display_columns
        ]
        .head(35)
        .to_string(index=False)
    )


def main():
    args = parse_arguments()
    validate_arguments(args)

    output_prefix = Path(
        args.out_prefix
    )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("[INFO] split_role:", args.split_role)
    print("[INFO] lambda_dc:", args.lambda_dc)
    print(
        "[INFO] minimum damage thresholds:",
        {
            "structural": (
                args.min_struct_damage
            ),
            "character": (
                args.min_char_damage
            ),
        },
    )
    print("=" * 100)

    if args.row_metrics_input:
        print(
            "[INFO] reusing row metrics:",
            args.row_metrics_input,
        )

        row_metrics = (
            load_and_enrich_existing_row_metrics(
                args.row_metrics_input,
                args,
            )
        )
    else:
        print(
            "[INFO] extracting metrics from:",
            args.input_path,
        )

        row_metrics = extract_row_metrics(
            args
        )

    row_metrics_path = (
        f"{args.out_prefix}"
        "_row_metrics_v2.jsonl"
    )

    save_jsonl(
        row_metrics,
        row_metrics_path,
    )

    print(
        "[SAVE]",
        row_metrics_path,
    )

    print(
        "[INFO] row perturbation types:",
        Counter(
            row["perturbation_type"]
            for row in row_metrics
        ),
    )

    source_dataframe = (
        aggregate_source_features(
            row_metrics,
            lambda_dc=args.lambda_dc,
        )
    )

    source_dataframe = check_pairing(
        source_dataframe
    )

    source_features_path = (
        f"{args.out_prefix}"
        "_source_features_v2.csv"
    )

    source_dataframe.to_csv(
        source_features_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[SAVE]",
        source_features_path,
    )

    print(
        "[INFO] source rows:",
        len(source_dataframe),
    )

    print(
        "[INFO] perturbation counts per source:",
        Counter(
            source_dataframe[
                "n_perturbations"
            ]
        ),
    )

    single_report = single_feature_report(
        source_dataframe,
        split_role=args.split_role,
    )

    single_report_path = (
        f"{args.out_prefix}"
        "_single_feature_report_v2.csv"
    )

    single_report.to_csv(
        single_report_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[SAVE]",
        single_report_path,
    )

    print_top_single_features(
        single_report,
        args.split_role,
    )

    if args.split_role in {"train", "dev"}:
        lambda_report = dc_lambda_scan(
            source_dataframe
        )

        lambda_report_path = (
            f"{args.out_prefix}"
            "_dc_lambda_scan_v2.csv"
        )

        lambda_report.to_csv(
            lambda_report_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "[SAVE]",
            lambda_report_path,
        )

        print()
        print("=" * 100)
        print("Top DC lambda settings")
        print("=" * 100)

        print(
            lambda_report
            .head(10)
            .to_string(index=False)
        )

        correlation, high_pairs = (
            feature_correlation_reports(
                source_dataframe,
                threshold=(
                    args.correlation_threshold
                ),
            )
        )

        correlation_path = (
            f"{args.out_prefix}"
            "_candidate_spearman_v2.csv"
        )

        high_pairs_path = (
            f"{args.out_prefix}"
            "_high_correlation_pairs_v2.csv"
        )

        correlation.to_csv(
            correlation_path,
            encoding="utf-8-sig",
        )

        high_pairs.to_csv(
            high_pairs_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "[SAVE]",
            correlation_path,
        )

        print(
            "[SAVE]",
            high_pairs_path,
        )

    if (
        args.split_role in {"train", "dev"}
        and not args.skip_cv
    ):
        cv_report = grouped_cv_report(
            source_dataframe,
            random_state=args.random_state,
            n_splits=args.cv_splits,
        )

        cv_report_path = (
            f"{args.out_prefix}"
            "_grouped_cv_report_v2.csv"
        )

        cv_report.to_csv(
            cv_report_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "[SAVE]",
            cv_report_path,
        )

        print()
        print("=" * 100)
        print("Grouped cross-validation")
        print("=" * 100)

        print(
            cv_report.to_string(
                index=False
            )
        )

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()