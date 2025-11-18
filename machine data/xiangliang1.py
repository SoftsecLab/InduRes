import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# 初始化
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


def rewrite_variants_clean(sentence, num_return_sequences=8):
    prompt = (
        f"请将下面这句话改写成 {num_return_sequences} 条不同的表达，"
        "保持语义一致：\n"
        f"{sentence}\n改写："
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        max_new_tokens=128,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text.replace(prompt, "").strip()
    cleaned = []
    for raw in text.splitlines():
        line = re.sub(r'^(?:\(?\d+\)?[\.、:：]?\s*)', '', raw.strip())
        # 只保留以句号/问号/感叹号结尾的完整句
        if re.match(r'.+[。！？!?]$', line):
            cleaned.append(line)
        if len(cleaned) >= num_return_sequences:
            break
    return cleaned


# 处理并保存
input_path = "/home/wangl/sh/mucgec_filtered_rewrite_path.jsonl"
output_path = "/home/wangl/sh/mucgec_rewrite_complete8.jsonl"

# 读取已经处理过的句子的索引（从文件中）
processed_indices = set()
with open(output_path, "r", encoding="utf-8") as f_out:
    for line in f_out:
        processed_data = json.loads(line)
        processed_indices.add(processed_data["source"])  # 假设每个条目中 "source" 是唯一标识

# 继续处理没有处理过的句子
with open(input_path, "r", encoding="utf-8") as f_in, \
        open(output_path, "a", encoding="utf-8") as f_out:  # 使用 "a" 模式来追加内容

    for idx, line in enumerate(tqdm(f_in, desc="Processing"), 1):
        src = json.loads(line)["source"].strip()

        if src in processed_indices:  # 如果已经处理过，跳过该句子
            continue

        print(f"\n=== 样本 {idx} 原句 ===\n{src}\n")
        variants = rewrite_variants_clean(src, 8)
        for i, v in enumerate(variants, 1):
            print(f"{i}. {v}")

        f_out.write(json.dumps({"source": src, "rewrites": variants}, ensure_ascii=False) + "\n")
        f_out.flush()
