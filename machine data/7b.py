import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM


# —————— 模型初始化 ——————
model_dir = "/home/wangl/sh/QwenQwen-7B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, trust_remote_code=True, local_files_only=True
)
model = AutoGPTQForCausalLM.from_quantized(
    model_dir,
    use_safetensors=True,
    device_map={"": "cuda:0"},
    trust_remote_code=True
).eval()

# —————— 改写函数（机器风格提示） ——————
def rewrite_variants_clean(sentence, num_return_sequences=1):
    prompt = (
        f"<机器助手>：请以“简洁精准、条理清晰”的机器风格，"
        f"对下面这句话进行改写，共生成 {num_return_sequences} 条不同表达：\n"
        f"{sentence}\n<改写开始>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        max_new_tokens=128,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )
    # 解码并清洗
    text_all = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    cleaned = []
    for raw in text_all:
        content = raw.replace(prompt, "").strip()
        for line in content.splitlines():
            line = re.sub(r'^(?:\(?\d+\)?[\.、:：]?\s*)', '', line.strip())
            if re.match(r'.+[。！？!?]$', line):
                cleaned.append(line)
            if len(cleaned) >= num_return_sequences:
                break
        if len(cleaned) >= num_return_sequences:
            break
    return cleaned

# —————— 路径定义 ——————
# 原始文件（包含已有改写，作为输入逐行读取）
output_path = "/home/wangl/sh/11qianwen_filled.jsonl"
# 目标文件（追加写入，断点续跑）
filled_output_path = "/home/wangl/sh/50qianwen_filled.jsonl"

# —————— 统计总行数，用于 tqdm ——————
with open(output_path, "r", encoding="utf-8") as fp:
    total_lines = sum(1 for _ in fp)

# —————— 断点续跑：先把已处理的 source 读入集合 ——————
processed = set()
if os.path.exists(filled_output_path):
    with open(filled_output_path, "r", encoding="utf-8") as f_done:
        for ln in f_done:
            try:
                data = json.loads(ln)
                processed.add(data["source"])
            except:
                continue

# —————— 主循环：追加模式写入，边运行边保存 ——————
with open(output_path, "r", encoding="utf-8") as fin, \
     open(filled_output_path, "a", encoding="utf-8") as fout:

    for idx, line in enumerate(tqdm(fin, total=total_lines, desc="补全改写", unit="条"), start=1):
        data = json.loads(line)
        src = data.get("source", "").strip()

        # 跳过已处理过的条目
        if src in processed:
            continue

        existing = data.get("rewrites", [])
        # 目标总数 50 条，计算还需生成多少
        need = max(0, 50 - len(existing))
        if need > 0:
            # 生成缺少的改写
            new_variants = rewrite_variants_clean(src, num_return_sequences=need)
            # 终端打印
            print(f"\n>>> 样本 {idx} 追加 {need} 条改写 <<<\n原文：{src}\n")
            for i, v in enumerate(new_variants, 1):
                print(f"{i}. {v}")
            # 追加到 existing
            existing.extend(new_variants)
            data["rewrites"] = existing

        # 写入目标文件并立即 flush
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        fout.flush()
        # 标记为已处理
        processed.add(src)

print(f"\n✅ 全部处理完成，结果保存在：{filled_output_path}")
