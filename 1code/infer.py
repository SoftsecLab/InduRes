import os
import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===============================
# 路径
# ===============================
output_dir = "/home/chy/qwen05b_incremental"   # 你训练好的模型保存路径
files = [
    "/home/chy/new/data/ec_law.tsv",
    "/home/chy/new/data/ec_med.tsv",
    "/home/chy/new/data/grammar.tsv",
    "/home/chy/new/data/lemon_car.tsv",
    "/home/chy/new/data/lemon_gam.tsv",
    "/home/chy/new/data/lemon_mec.tsv",
    "/home/chy/new/data/lemon_new.tsv",
    "/home/chy/new/data/medical_csc.tsv",
    "/home/chy/new/data/TextProofreadingCompetition.tsv",
]

# ===============================
# 加载数据
# ===============================
dfs = [pd.read_csv(f, sep="\t") for f in files]
df = pd.concat(dfs, ignore_index=True)

data = [
    {
        "source": row.get("source", ""),
        "target": row.get("target", ""),
        "positive": row.get("positive", ""),
        "negative": row.get("negative", ""),
        "type": row.get("type", "")
    }
    for _, row in df.iterrows()
    if pd.notna(row.get("source")) and pd.notna(row.get("target"))
]

print(f"✅ 数据加载成功，共 {len(data)} 条")

# ===============================
# 只抽取 type == negative 的
# ===============================
negative_data = [d for d in data if str(d.get("type", "")).lower() == "negative"]
samples = random.sample(negative_data, min(50, len(negative_data)))
print(f"✅ 已随机抽取 {len(samples)} 条 type=negative 样本用于测试\n")

# ===============================
# 加载模型和 tokenizer
# ===============================
tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto")



def clean_prediction(text: str) -> str:
    if "答案：" in text:
        pred = text.split("答案：", 1)[1]
    else:
        pred = text

    # 去掉多余的换行和空格
    pred = pred.replace("\n", " ").strip()

    # 只取到第一个句号/问号/感叹号为止
    import re
    m = re.split(r"[。？！]", pred)
    if m:
        return m[0].strip()
    return pred.strip()

# ===============================
# 推理函数（beam search + 去重复）
# ===============================
def generate_correction(sentence, max_new_tokens=64):
    prompt = f"请纠正这句话：{sentence}\n答案："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,                 # beam search 更稳定
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_prediction(result)

# ===============================
# 运行预测并显示结果
# ===============================
for i, ex in enumerate(samples, 1):
    src = ex["source"]
    tgt = ex["target"]
    pos = ex["positive"]
    neg = ex["negative"]
    typ = ex["type"]
    pred = generate_correction(src)
    print(f"==== 样本 {i} ====")
    print(f"原句       : {src}")
    print(f"标准答案   : {tgt}")
    print(f"正样本     : {pos}")
    print(f"负样本     : {neg}")
    print(f"type       : {typ}")
    print(f"模型预测   : {pred}\n")
