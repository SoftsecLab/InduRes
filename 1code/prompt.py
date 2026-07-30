import os
import json
from collections import defaultdict, Counter


# ============================================================
# 1. 路径配置
# ============================================================

TASK_DIR = "/home/chy/1/data/hc3_processed/rewrite_tasks"

NEUTRAL_TASK_PATHS = {
    "train700": os.path.join(
        TASK_DIR,
        "rewrite_tasks_train700_neutral.jsonl"
    ),
    "dev100": os.path.join(
        TASK_DIR,
        "rewrite_tasks_dev100_neutral.jsonl"
    ),
    "test200": os.path.join(
        TASK_DIR,
        "rewrite_tasks_test200_neutral.jsonl"
    ),
}

N_REWRITES = 3

STYLE_TYPES = [
    "human_style",
    "machine_style",
]

os.makedirs(TASK_DIR, exist_ok=True)


# ============================================================
# 2. Human-style prompt
# ============================================================

def human_style_prompt(text, n=3):
    return f"""请在保持原意、事实信息和信息完整性不变的前提下，对下面这段中文文本生成 {n} 条具有自然人工表达风格的中文改写。

风格要求：
1. 表达应自然、流畅，更接近真实用户日常写出的回答。
2. 避免高度模板化、机械化或过度规整的表达。
3. 可以调整句式、语序、措辞和衔接方式，使语言更灵活自然。
4. 可以使用自然的语气和表达习惯，但不要刻意制造错别字、语病或网络化表达。
5. 只能改变表达方式，不能对原文进行概括、摘要或压缩。
6. 必须保留原文中的全部事实、条件、建议、步骤和关键信息。
7. 改写后的文本长度应与原文大致相近，不要明显缩短或扩写。
8. 如果原文包含多个要点，可以将其自然地组织成连续段落，但不得合并、遗漏或改变任何要点。
9. 每条改写应使用不同的表达方式，但都必须保持相同的信息覆盖范围。
10. 每条改写应为一个连续的文本段落，不要写成标题、编号列表或分点内容。
11. 必须严格输出 JSON 字符串数组。
12. 数组中的每个元素必须是字符串，不能是对象，不能使用键值对。
13. 不要添加解释、标题、编号或 Markdown 代码块。

正确输出格式示例：
["改写文本1", "改写文本2", "改写文本3"]

错误输出格式示例：
[{{"改写文本1"}}, {{"改写文本2"}}]

原文：
{text}
"""


# ============================================================
# 3. Machine-style prompt
# ============================================================

def machine_style_prompt(text, n=3):
    return f"""请在保持原意、事实信息和信息完整性不变的前提下，对下面这段中文文本生成 {n} 条具有典型机器生成表达风格的中文改写。

风格要求：
1. 表达应规范、完整、结构清晰，并具有较强的书面化特征。
2. 使用明确的逻辑关系、稳定的句式结构和清晰的衔接方式。
3. 表达可以更加系统、严谨和模板化，使其接近大语言模型生成的回答风格。
4. 可以调整句式、语序、措辞和逻辑连接方式，但不能增加原文没有的新事实。
5. 只能改变表达方式，不能对原文进行概括、摘要或压缩。
6. 必须保留原文中的全部事实、条件、建议、步骤和关键信息。
7. 改写后的文本长度应与原文大致相近，不要明显缩短或扩写。
8. 如果原文包含多个要点，应以清晰、系统的方式组织这些要点，但不得合并、遗漏或改变任何要点。
9. 每条改写应使用不同的表达方式，但都必须保持相同的信息覆盖范围和机器生成风格。
10. 每条改写应为一个连续的文本段落，不要写成标题、编号列表或分点内容。
11. 必须严格输出 JSON 字符串数组。
12. 数组中的每个元素必须是字符串，不能是对象，不能使用键值对。
13. 不要添加解释、标题、编号或 Markdown 代码块。

正确输出格式示例：
["改写文本1", "改写文本2", "改写文本3"]

错误输出格式示例：
[{{"改写文本1"}}, {{"改写文本2"}}]

原文：
{text}
"""


PROMPT_FUNCTIONS = {
    "human_style": human_style_prompt,
    "machine_style": machine_style_prompt,
}


# ============================================================
# 4. JSONL 读写函数
# ============================================================

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(
                    f"Failed to parse JSON at {path}, line {line_no}: {e}"
                )

    return rows


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# 5. 检查 neutral 任务文件
# ============================================================

def validate_neutral_tasks(tasks, split_name):
    required_fields = [
        "task_id",
        "sample_id",
        "topic_id",
        "source_label",
        "label",
        "source_text",
    ]

    topic_groups = defaultdict(list)
    sample_ids = []
    task_ids = []

    for index, task in enumerate(tasks):
        for field in required_fields:
            if field not in task:
                raise ValueError(
                    f"{split_name}: task index {index} missing field: {field}"
                )

        source_label = task["source_label"]
        label = task["label"]

        if source_label not in {"human", "machine"}:
            raise ValueError(
                f"{split_name}: invalid source_label={source_label}"
            )

        expected_label = 0 if source_label == "human" else 1

        if label != expected_label:
            raise ValueError(
                f"{split_name}: sample_id={task['sample_id']} has "
                f"source_label={source_label}, label={label}, "
                f"expected label={expected_label}"
            )

        if not str(task["source_text"]).strip():
            raise ValueError(
                f"{split_name}: empty source_text for {task['sample_id']}"
            )

        topic_groups[task["topic_id"]].append(task)
        sample_ids.append(task["sample_id"])
        task_ids.append(task["task_id"])

    duplicate_sample_ids = [
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    ]

    if duplicate_sample_ids:
        raise ValueError(
            f"{split_name}: duplicate sample_ids found: "
            f"{duplicate_sample_ids[:10]}"
        )

    duplicate_task_ids = [
        task_id
        for task_id, count in Counter(task_ids).items()
        if count > 1
    ]

    if duplicate_task_ids:
        raise ValueError(
            f"{split_name}: duplicate task_ids found: "
            f"{duplicate_task_ids[:10]}"
        )

    invalid_topics = []

    for topic_id, items in topic_groups.items():
        labels = {item["source_label"] for item in items}

        if len(items) != 2 or labels != {"human", "machine"}:
            invalid_topics.append({
                "topic_id": topic_id,
                "n_items": len(items),
                "source_labels": sorted(labels),
            })

    if invalid_topics:
        raise ValueError(
            f"{split_name}: invalid paired topics found, "
            f"examples={invalid_topics[:5]}"
        )

    return {
        "n_tasks": len(tasks),
        "n_topics": len(topic_groups),
        "n_human": sum(
            task["source_label"] == "human"
            for task in tasks
        ),
        "n_machine": sum(
            task["source_label"] == "machine"
            for task in tasks
        ),
    }


# ============================================================
# 6. 从 neutral task 派生 style-specific task
# ============================================================

def build_style_tasks(neutral_tasks, style_type):
    if style_type not in PROMPT_FUNCTIONS:
        raise ValueError(f"Unknown style type: {style_type}")

    prompt_function = PROMPT_FUNCTIONS[style_type]
    style_tasks = []

    for neutral_task in neutral_tasks:
        # 创建副本，避免修改原始 neutral task
        task = dict(neutral_task)

        # 保存 neutral 条件下的任务 ID，方便后续对齐检查
        task["parent_task_id"] = neutral_task["task_id"]

        # 新任务 ID
        task["task_id"] = (
            f'{neutral_task["sample_id"]}_{style_type}'
        )

        # induction 类型
        task["prompt_type"] = style_type
        task["induction_style"] = style_type

        # 每个 source 生成 3 条 rewrite
        task["n_rewrites"] = N_REWRITES

        # 替换为相应的 style-specific prompt
        task["prompt"] = prompt_function(
            neutral_task["source_text"],
            N_REWRITES,
        )

        # 标签完全继承 source provenance
        expected_label = (
            0 if task["source_label"] == "human" else 1
        )

        if task["label"] != expected_label:
            raise ValueError(
                f'Label mismatch for sample_id={task["sample_id"]}: '
                f'source_label={task["source_label"]}, '
                f'label={task["label"]}'
            )

        style_tasks.append(task)

    return style_tasks


# ============================================================
# 7. 检查不同 induction 条件是否严格对齐
# ============================================================

def validate_alignment(neutral_tasks, style_tasks, style_type):
    if len(neutral_tasks) != len(style_tasks):
        raise ValueError(
            f"{style_type}: task number mismatch: "
            f"neutral={len(neutral_tasks)}, "
            f"style={len(style_tasks)}"
        )

    compare_fields = [
        "sample_id",
        "topic_id",
        "split",
        "source_label",
        "label",
        "source_text",
    ]

    for index, (neutral_task, style_task) in enumerate(
        zip(neutral_tasks, style_tasks)
    ):
        for field in compare_fields:
            if neutral_task.get(field) != style_task.get(field):
                raise ValueError(
                    f"{style_type}: alignment mismatch at index={index}, "
                    f"field={field}"
                )

        expected_task_id = (
            f'{neutral_task["sample_id"]}_{style_type}'
        )

        if style_task["task_id"] != expected_task_id:
            raise ValueError(
                f"{style_type}: unexpected task_id at index={index}"
            )


# ============================================================
# 8. 主程序
# ============================================================

def main():
    print("=" * 90)
    print("Building style-specific rewrite tasks")
    print("=" * 90)

    for split_name, neutral_path in NEUTRAL_TASK_PATHS.items():
        if not os.path.exists(neutral_path):
            raise FileNotFoundError(
                f"Neutral task file not found: {neutral_path}"
            )

        neutral_tasks = load_jsonl(neutral_path)

        stats = validate_neutral_tasks(
            tasks=neutral_tasks,
            split_name=split_name,
        )

        print()
        print("-" * 90)
        print(f"[SPLIT] {split_name}")
        print(f"[INPUT] {neutral_path}")
        print(f"[TOPICS] {stats['n_topics']}")
        print(f"[SOURCE TASKS] {stats['n_tasks']}")
        print(f"[HUMAN SOURCES] {stats['n_human']}")
        print(f"[MACHINE SOURCES] {stats['n_machine']}")

        for style_type in STYLE_TYPES:
            style_tasks = build_style_tasks(
                neutral_tasks=neutral_tasks,
                style_type=style_type,
            )

            validate_alignment(
                neutral_tasks=neutral_tasks,
                style_tasks=style_tasks,
                style_type=style_type,
            )

            output_path = os.path.join(
                TASK_DIR,
                f"rewrite_tasks_{split_name}_{style_type}.jsonl",
            )

            save_jsonl(style_tasks, output_path)

            print()
            print(f"[STYLE] {style_type}")
            print(f"[TASKS] {len(style_tasks)}")
            print(
                f"[EXPECTED REWRITES] "
                f"{len(style_tasks) * N_REWRITES}"
            )
            print(f"[SAVED] {output_path}")

    print()
    print("=" * 90)
    print("[DONE] All style-specific task files were generated.")
    print("=" * 90)


if __name__ == "__main__":
    main()