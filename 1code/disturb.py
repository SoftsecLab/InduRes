#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
复现 dev100 的原始扰动流程，并支持 train/dev/test 任意 split。

保持不变的实验设计
------------------
1. Stanza dependency delete:
   deprel in {advmod, amod, obj}，随机删除长度 > 1 的词。
2. Connective delete:
   从原代码给定连接词表中随机删除一个。
3. Punctuation:
   从固定标点替换对中随机选择。
4. BERT lexical replace:
   使用 bert-base-chinese MLM 替换一个汉字。
5. 每个 topic 共用同一个五槽 plan：
   dependency_delete, connective_delete, punctuation,
   lexical_replace, random choice of four types.
6. random.seed(42), torch.manual_seed(42)。

后续纠错脚本会删除 lexical_replace，因此正式路径中每个 source
最终保留 3 或 4 个扰动，且同一 topic 的 human/machine 数量一致。
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import stanza
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def build_stanza_pipeline(stanza_dir, stanza_package):
    return stanza.Pipeline(
        lang="zh",
        processors="tokenize,pos,lemma,depparse",
        dir=stanza_dir,
        package=stanza_package,
        download_method=None,
        verbose=False,
    )


def dependency_delete(text, nlp):
    document = nlp(text)
    words = []

    for sentence in document.sentences:
        words.extend(sentence.words)

    candidates = []

    for word in words:
        if word.deprel in ["advmod", "amod", "obj"]:
            if len(word.text) > 1:
                candidates.append(word.text)

    if not candidates:
        return text, False

    target = random.choice(candidates)

    new_text = text.replace(
        target,
        "",
        1,
    )

    if new_text != text:
        return new_text, True

    return text, False


def connective_delete(text):
    connectives = [
        "但是",
        "然而",
        "所以",
        "因此",
        "然后",
        "其实",
        "另外",
        "同时",
        "可能",
        "一般来说",
        "总的来说",
    ]

    candidates = [
        connective
        for connective in connectives
        if connective in text
    ]

    if not candidates:
        return text, False

    target = random.choice(candidates)

    new_text = text.replace(
        target,
        "",
        1,
    )

    return new_text, True


def punctuation_perturb(text):
    pairs = [
        ("。", "，"),
        ("，", "。"),
        ("！", "。"),
        ("？", "。"),
    ]

    candidates = [
        pair
        for pair in pairs
        if pair[0] in text
    ]

    if not candidates:
        return text, False

    source_punctuation, target_punctuation = random.choice(
        candidates
    )

    new_text = text.replace(
        source_punctuation,
        target_punctuation,
        1,
    )

    return new_text, True


def bert_replace(
    text,
    model,
    tokenizer,
    top_k=10,
):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    input_ids = encoded["input_ids"].to(DEVICE)

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0]
    )

    candidates = []

    for index, token in enumerate(tokens):
        if (
            len(token) == 1
            and "\u4e00" <= token <= "\u9fff"
            and token not in ["的", "了", "是", "我", "你"]
        ):
            candidates.append(index)

    if not candidates:
        return text, False

    position = random.choice(candidates)
    masked_ids = input_ids.clone()
    masked_ids[0, position] = tokenizer.mask_token_id

    with torch.no_grad():
        output = model(masked_ids)

    probabilities = torch.softmax(
        output.logits[0, position],
        dim=-1,
    )

    values, indices = torch.topk(
        probabilities,
        k=top_k,
    )

    new_token_id = random.choices(
        indices.tolist(),
        weights=values.tolist(),
        k=1,
    )[0]

    masked_ids[0, position] = new_token_id

    new_tokens = tokenizer.convert_ids_to_tokens(
        masked_ids[0]
    )

    result = []

    for token in new_tokens:
        if token in ["[CLS]", "[SEP]"]:
            continue

        result.append(
            token.replace("##", "")
        )

    new_text = "".join(result)

    if "[UNK]" in new_text:
        return text, False

    if new_text != text:
        return new_text, True

    return text, False


def apply_perturb(
    text,
    perturb_type,
    nlp,
    model,
    tokenizer,
):
    if perturb_type == "dependency_delete":
        return dependency_delete(text, nlp)

    if perturb_type == "connective_delete":
        return connective_delete(text)

    if perturb_type == "punctuation":
        return punctuation_perturb(text)

    if perturb_type == "lexical_replace":
        return bert_replace(
            text,
            model,
            tokenizer,
        )

    raise ValueError(
        f"未知扰动类型：{perturb_type}"
    )


def generate_plan():
    perturb_types = [
        "dependency_delete",
        "connective_delete",
        "punctuation",
        "lexical_replace",
    ]

    return [
        "dependency_delete",
        "connective_delete",
        "punctuation",
        "lexical_replace",
        random.choice(perturb_types),
    ]


def validate_input_pairs(data):
    topics = defaultdict(list)

    for item in data:
        topics[item["topic_id"]].append(item)

    invalid_topics = []

    for topic_id, items in topics.items():
        label_counts = Counter(
            str(item["source_label"]).strip().lower()
            for item in items
        )

        if (
            len(items) != 2
            or label_counts["human"] != 1
            or label_counts["machine"] != 1
        ):
            invalid_topics.append(
                (topic_id, dict(label_counts), len(items))
            )

    if invalid_topics:
        raise ValueError(
            "输入中存在非严格配对 topic，示例："
            f"{invalid_topics[:20]}"
        )

    return topics


def validate_output_sync(outputs):
    signatures = defaultdict(
        lambda: defaultdict(list)
    )

    for row in outputs:
        signatures[
            row["topic_id"]
        ][
            row["source_label"]
        ].append(
            (
                int(row["perturb_id"]),
                row["perturb_type"],
            )
        )

    mismatches = []

    for topic_id, by_label in signatures.items():
        if by_label["human"] != by_label["machine"]:
            mismatches.append(topic_id)

    if mismatches:
        raise RuntimeError(
            "human/machine 扰动计划不同："
            f"{mismatches[:20]}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        required=True,
    )

    parser.add_argument(
        "--output_path",
        required=True,
    )

    parser.add_argument(
        "--stanza_dir",
        default="/home/chy/1/sta",
    )

    parser.add_argument(
        "--stanza_package",
        default="gsdsimp",
    )

    parser.add_argument(
        "--bert_path",
        default="/home/chy/1/bert-base-chinese",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[INFO] device:", DEVICE)
    print("[INFO] loading Stanza")

    nlp = build_stanza_pipeline(
        args.stanza_dir,
        args.stanza_package,
    )

    print("[INFO] loading BERT tokenizer/model")

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_path,
        local_files_only=True,
    )

    model = BertForMaskedLM.from_pretrained(
        args.bert_path,
        local_files_only=True,
    )

    model.to(DEVICE)
    model.eval()

    data = load_jsonl(args.input_path)
    topics = validate_input_pairs(data)

    print("[INFO] input rows:", len(data))
    print("[INFO] topics:", len(topics))

    outputs = []
    plan_counts = Counter()
    success_counts = Counter()

    for topic_id, items in tqdm(
        topics.items(),
        desc="Perturb pairs",
    ):
        plan = generate_plan()
        plan_counts.update(plan)

        for item in items:
            source_text = item["source_text"]

            for perturb_id, perturb_type in enumerate(
                plan,
                start=1,
            ):
                new_text, success = apply_perturb(
                    source_text,
                    perturb_type,
                    nlp,
                    model,
                    tokenizer,
                )

                row = item.copy()
                row["perturb_id"] = perturb_id
                row["perturb_type"] = perturb_type
                row["perturb_success"] = bool(success)
                row["perturbed_text"] = new_text
                row["perturb_seed"] = args.seed
                row["perturb_generator"] = (
                    "original_stanza_bert_sync"
                )

                outputs.append(row)

                success_counts[
                    (
                        perturb_type,
                        bool(success),
                    )
                ] += 1

    validate_output_sync(outputs)
    save_jsonl(outputs, args.output_path)

    allowed_types = {
        "dependency_delete",
        "connective_delete",
        "punctuation",
    }

    after_filter_count = sum(
        row["perturb_type"] in allowed_types
        for row in outputs
    )

    per_source_after_filter = Counter()
    valid_by_sample = defaultdict(int)

    for row in outputs:
        if row["perturb_type"] in allowed_types:
            valid_by_sample[row["sample_id"]] += 1

    per_source_after_filter.update(
        valid_by_sample.values()
    )

    print("[INFO] plan type counts:", dict(plan_counts))
    print(
        "[INFO] success counts:",
        {
            f"{ptype}:{success}": count
            for (ptype, success), count
            in success_counts.items()
        },
    )
    print("[INFO] output rows:", len(outputs))
    print(
        "[INFO] rows after lexical filter:",
        after_filter_count,
    )
    print(
        "[INFO] retained perturbations per source:",
        dict(per_source_after_filter),
    )
    print("[SAVE]", args.output_path)


if __name__ == "__main__":
    main()