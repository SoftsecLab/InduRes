#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
复现 dev100 的原始 Qwen 扰动修正流程，并支持任意 split。

保持不变：
- 原始 prompt；
- 仅保留 dependency_delete/connective_delete/punctuation；
- do_sample=False；
- repetition_penalty=1.05；
- max_new_tokens=256；
- edit_ratio > 0.08 时回退为 perturbed_text；
- 扰动失败时直接保留 perturbed_text。

修复：
1. 每轮显式初始化 filtered=False，避免上一条记录的 filtered
   值泄漏到下一条记录。
2. 支持断点续跑。
3. 最终去重并按照输入顺序保存。
4. 检查 human/machine 配对数量和 3/4 条分布。
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from Levenshtein import distance
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALLOWED_TYPES = {
    "dependency_delete",
    "connective_delete",
    "punctuation",
}


def load_jsonl(path):
    rows = []

    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as exc:
                print(
                    f"[WARN] 第 {line_number} 行 JSON 错误：{exc}"
                )

    return rows


def append_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def save_jsonl(rows, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    os.replace(temporary, target)


def task_key(row):
    return "::".join(
        [
            str(row.get("sample_id", "")),
            str(row.get("perturb_id", "")),
            str(row.get("perturb_type", "")),
        ]
    )


def build_prompt(text):
    return f"""
下面文本经过人工加入局部扰动。

你的任务是恢复扰动，而不是改写。

严格要求：

1. 只修复明显由扰动导致的问题；
2. 保留原作者表达方式；
3. 保留原作者语气；
4. 保留原句结构；
5. 保留口语、不规范表达。

允许：
- 恢复删除内容；
- 修复错误标点；
- 修复明显语法破坏。

禁止：
- 润色；
- 改写；
- 增加信息；
- 删除信息；
- 改变表达风格。

如果无法确定存在错误，则保持原文。

只输出恢复后的文本，不要解释。

输入：

{text}

恢复文本：
"""


def edit_ratio(text_a, text_b):
    if len(text_a) == 0:
        return 1.0

    return distance(text_a, text_b) / len(text_a)


def clean_result(result):
    result = str(result or "").strip()

    prefixes = [
        "恢复文本：",
        "恢复后的文本：",
        "最终文本：",
    ]

    for prefix in prefixes:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
            break

    return result


def correct_text(
    text,
    tokenizer,
    model,
):
    messages = [
        {
            "role": "user",
            "content": build_prompt(text),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    result = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return clean_result(result)


def normalize_existing_rows(rows):
    latest = {}

    for row in rows:
        latest[task_key(row)] = row

    return latest


def validate_final_rows(rows):
    source_counts = Counter(
        row["source_label"]
        for row in rows
    )

    type_counts = Counter(
        row["perturb_type"]
        for row in rows
    )

    grouped_sources = defaultdict(int)

    for row in rows:
        grouped_sources[
            row["sample_id"]
        ] += 1

    per_source_counts = Counter(
        grouped_sources.values()
    )

    grouped_topics = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        grouped_topics[
            row["topic_id"]
        ][
            row["source_label"]
        ].append(
            (
                int(row["perturb_id"]),
                row["perturb_type"],
            )
        )

    mismatched = []

    for topic_id, by_label in grouped_topics.items():
        human_signature = sorted(
            by_label["human"]
        )
        machine_signature = sorted(
            by_label["machine"]
        )

        if human_signature != machine_signature:
            mismatched.append(topic_id)

    print("[CHECK] source labels:", dict(source_counts))
    print("[CHECK] perturb types:", dict(type_counts))
    print(
        "[CHECK] perturbations per source:",
        dict(per_source_counts),
    )
    print(
        "[CHECK] mismatched paired topics:",
        len(mismatched),
    )

    if source_counts["human"] != source_counts["machine"]:
        raise RuntimeError(
            "human/machine 最终行数不平衡。"
        )

    if mismatched:
        raise RuntimeError(
            "存在不同步的 human/machine 扰动："
            f"{mismatched[:20]}"
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
        "--model_path",
        default=(
            "/home/share/models/"
            "qwen2.5-7b-instruct"
        ),
    )

    parser.add_argument(
        "--max_rows",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    print("[INFO] device:", DEVICE)

    all_input_rows = load_jsonl(
        args.input_path
    )

    data = [
        row
        for row in all_input_rows
        if row.get("perturb_type") in ALLOWED_TYPES
    ]

    if args.max_rows > 0:
        data = data[:args.max_rows]

    existing_rows = load_jsonl(
        args.output_path
    )

    existing_latest = normalize_existing_rows(
        existing_rows
    )

    completed_keys = set(
        existing_latest
    )

    pending = [
        row
        for row in data
        if task_key(row) not in completed_keys
    ]

    print("[INFO] original rows:", len(all_input_rows))
    print("[INFO] after perturb filter:", len(data))
    print("[INFO] existing completed:", len(completed_keys))
    print("[INFO] pending:", len(pending))

    if pending:
        print("[INFO] loading tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        print("[INFO] loading Qwen")

        if torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        model.eval()

        for item in tqdm(
            pending,
            desc="Qwen correction",
        ):
            perturbed = item["perturbed_text"]

            # 必须每条记录重新初始化，避免 filtered 状态串行污染。
            filtered = False

            if item.get("perturb_success", True) is False:
                corrected = perturbed
            else:
                candidate = correct_text(
                    perturbed,
                    tokenizer,
                    model,
                )

                ratio = edit_ratio(
                    perturbed,
                    candidate,
                )

                if ratio > 0.08:
                    corrected = perturbed
                    filtered = True
                else:
                    corrected = candidate
                    filtered = False

            new_row = item.copy()

            new_row["corrected_text"] = corrected
            new_row["correction_changed"] = (
                corrected != perturbed
            )
            new_row["edit_ratio"] = edit_ratio(
                perturbed,
                corrected,
            )
            new_row["filtered"] = filtered

            append_jsonl(
                args.output_path,
                [new_row],
            )

    # 断点续跑后重新读取，按输入顺序去重整理。
    final_latest = normalize_existing_rows(
        load_jsonl(args.output_path)
    )

    final_rows = []

    for input_row in data:
        key = task_key(input_row)

        if key not in final_latest:
            raise RuntimeError(
                f"任务尚未生成：{key}"
            )

        final_rows.append(
            final_latest[key]
        )

    validate_final_rows(final_rows)
    save_jsonl(final_rows, args.output_path)

    changed = sum(
        bool(row["correction_changed"])
        for row in final_rows
    )

    average_edit_ratio = (
        sum(
            float(row["edit_ratio"])
            for row in final_rows
        )
        / len(final_rows)
        if final_rows
        else 0.0
    )

    filtered_count = sum(
        bool(row.get("filtered", False))
        for row in final_rows
    )

    failed_perturb_count = sum(
        not bool(
            row.get("perturb_success", True)
        )
        for row in final_rows
    )

    print("========== Overall ==========")
    print("Total:", len(final_rows))
    print(
        "Correction rate:",
        changed / len(final_rows)
        if final_rows
        else 0.0,
    )
    print(
        "Average edit ratio:",
        average_edit_ratio,
    )
    print(
        "Over-rewrite fallback rows:",
        filtered_count,
    )
    print(
        "Failed perturbation rows:",
        failed_perturb_count,
    )
    print("[SAVE]", args.output_path)


if __name__ == "__main__":
    main()