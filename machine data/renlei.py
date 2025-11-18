import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================
# 1. 模型路径 & 加载
# ==========================
model_dir = "/home/wangl/sh/Yi-6B-Chat"  # 本地 Yi-6B-Chat 模型路径
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
).eval()


# ==========================
# 2. Few‐shot 改写函数（动态生成 need 条）
# ==========================
def generate_human_style_variants(source_sentence, example_rewrites, num_new):
    """
    few‐shot Prompt：用前两条 example_rewrites 给示例，
    生成 num_new 条“表达方式不同、意思相同”的自然中文改写。
    """
    ex1, ex2 = example_rewrites[:2]
    prompt = (
        "下面给出一句话及其两种人类写作风格的改写示例：\n\n"
        f"原句：{source_sentence}\n"
        f"示例改写1：{ex1}\n"
        f"示例改写2：{ex2}\n\n"
        f"请在此基础上，再为原句生成 {num_new} 条“表达方式不同但意思相同”的自然中文改写，"
        "每条都要像人自己写的一样，不要出现“AI”或“助手”字样。\n"
    )

    new_set = set()
    max_attempts = num_new * 8
    attempts = 0

    while attempts < max_attempts and len(new_set) < num_new:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            max_new_tokens=80,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = raw.replace(prompt, "").strip().splitlines()[0]
        text = re.sub(r'^(?:\(?\d+\)?[\.、:：]?\s*)', "", text)

        if re.match(r'.+?[。！？!?]$', text) \
           and text not in example_rewrites \
           and text not in new_set:
            new_set.add(text)

        attempts += 1

    return list(new_set)[:num_new]


# ==========================
# 3. 主流程：断点续跑 → 动态补齐到 50 条 → tqdm → 边运行边保存
# ==========================
TARGET_TOTAL = 50
input_path = "/home/wangl/sh/merged_human.jsonl"
output_path = "/home/wangl/sh/50renlei_rewrite_yi.jsonl"

# 记录已处理过的 source，方便断点续跑
processed = set()
try:
    with open(output_path, "r", encoding="utf-8") as fin:
        for ln in fin:
            processed.add(json.loads(ln)["source"])
except FileNotFoundError:
    pass

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "a", encoding="utf-8") as fout:

    for idx, ln in enumerate(tqdm(fin, desc="扩充到50条进行中", unit="条"), start=1):
        item = json.loads(ln)
        src = item["source"].strip()
        existing = item.get("rewrites", [])

        # 跳过已完成的
        if src in processed:
            continue

        # 至少需要两条示例改写才能 few‐shot
        if len(existing) < 2:
            print(f"[跳过] 样本 {idx} 示例改写不足两条：{src}")
            processed.add(src)
            continue

        # 计算还需多少条
        if len(existing) >= TARGET_TOTAL:
            combined = existing[:TARGET_TOTAL]
            print(f"[样本 {idx}] 已有 {len(existing)} 条，截取前 {TARGET_TOTAL} 条。")
        else:
            need = TARGET_TOTAL - len(existing)
            print("\n" + "=" * 50)
            print(f"样本 {idx} 原句：{src}")
            print(f"已有 {len(existing)} 条示例，需生成 {need} 条新改写：")
            new_rs = generate_human_style_variants(src, existing, num_new=need)

            for j, sent in enumerate(new_rs, start=1):
                print(f"  新{j}: {sent}")

            combined = existing + new_rs

        # 写入并立即 flush
        fout.write(json.dumps({"source": src, "rewrites": combined}, ensure_ascii=False) + "\n")
        fout.flush()
        processed.add(src)

print("\n✅ 全部样本已扩充至 50 条改写，处理完毕。")
