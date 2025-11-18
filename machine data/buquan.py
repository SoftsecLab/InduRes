import os
import re
import json
import torch
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

# —————— 改写函数 ——————
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

# —————— 路径定义 ——————
input_path = "/home/wangl/sh/mucgec_filtered_rewrite_path.jsonl"       # 原始源文件
output_path = "/home/wangl/sh/qianwen_filled.jsonl"                    # 现有包含空改写的文件
filled_output_path = "/home/wangl/sh/11qianwen_filled.jsonl"      # 补全后输出的新文件

# —————— 统计原文件行数，以便 tqdm 显示 ——————
with open(output_path, "r", encoding="utf-8") as fp:
    total_lines = sum(1 for _ in fp)

# —————— 遍历并补全空改写 ——————
with open(output_path, "r", encoding="utf-8") as fin, \
     open(filled_output_path, "w", encoding="utf-8") as fout:

    for idx, line in enumerate(tqdm(fin, total=total_lines, desc="补全空改写", unit="line"), 1):
        data = json.loads(line)
        src = data.get("source", "").strip()
        existing_rewrites = data.get("rewrites", [])

        # 如果 rewrites 为空列表或长度不足，则再次调用模型
        if not existing_rewrites or len(existing_rewrites) < 1:
            print(f"\n=== 样本 {idx} 原句 ===\n{src}\n")
            new_variants = rewrite_variants_clean(src, num_return_sequences=8)
            for i, v in enumerate(new_variants, 1):
                print(f"{i}. {v}")
            data["rewrites"] = new_variants  # 更新改写结果

        # 将（更新后的）条目写入新文件
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        fout.flush()

print(f"\n✅ 补全完成，结果已保存到：{filled_output_path}")

