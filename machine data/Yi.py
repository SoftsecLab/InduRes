import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径
model_dir = "/home/wangl/models/Yi-6B-Chat"

# 加载 tokenizer 和模型
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


# 多轮采样函数，每轮生成 1 条，共生成 8 条
def rewrite_variants_clean(sentence, num_return_sequences=8):
    prompt = (
        "请将下面这句话改写成意思相同但表达方式不同的一句话：\n"
        f"{sentence}\n改写："
    )
    cleaned = set()
    for _ in range(num_return_sequences * 2):  # 多采样防止重复或空
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            max_new_tokens=64,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = text.replace(prompt, "").strip()

        # 清理格式
        line = re.sub(r'^(?:\(?\d+\)?[\.、:：]?\s*)', '', text)
        if re.match(r'.+[。！？!?]$', line):
            cleaned.add(line)

        if len(cleaned) >= num_return_sequences:
            break

    return list(cleaned)[:num_return_sequences]


# 输入输出路径
input_path = "/home/wangl/data/mucgec_filtered_rewrite_path.jsonl"
output_path = "/home/wangl/data/mucgec_rewrite_complete8.jsonl"

# 已处理样本集合
processed_indices = set()
try:
    with open(output_path, "r", encoding="utf-8") as f_out:
        for line in f_out:
            processed_data = json.loads(line)
            processed_indices.add(processed_data["source"])
except FileNotFoundError:
    pass

# 主处理流程
with open(input_path, "r", encoding="utf-8") as f_in, \
        open(output_path, "a", encoding="utf-8") as f_out:
    for idx, line in enumerate(tqdm(f_in, desc="改写中"), 1):
        try:
            src = json.loads(line)["source"].strip()
            if src in processed_indices:
                continue

            print(f"\n=== 样本 {idx} 原句 ===\n{src}\n")
            variants = rewrite_variants_clean(src, 8)
            for i, v in enumerate(variants, 1):
                print(f"{i}. {v}")

            f_out.write(json.dumps({"source": src, "rewrites": variants}, ensure_ascii=False) + "\n")
            f_out.flush()

        except Exception as e:
            print(f"【出错跳过】第 {idx} 条样本错误：{e}")
            continue
