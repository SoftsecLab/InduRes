import os
import re
import json
from tqdm import tqdm
from openai import OpenAI

# ========== 配置 ==========
API_KEY = "你的_API_KEY"   # <<< 替换成你自己的 OpenAI API key
MODEL = "gpt-4o-mini"     # 推荐使用 gpt-4o-mini，便宜且够用
BATCH_SIZE = 5            # 每次生成几条，避免请求过大
TARGET_COUNT = 50         # 每条文本需要的改写数

client = OpenAI(api_key=API_KEY)

# ========== 改写函数 ==========
def rewrite_variants_clean(sentence, num_return_sequences=1):
    """
    调用 OpenAI GPT API 生成改写，返回一个 list
    """
    prompt = (
        f"<机器助手>：请以“简洁精准、条理清晰”的机器风格，"
        f"对下面这句话进行改写，共生成 {num_return_sequences} 条不同表达：\n"
        f"{sentence}\n<改写开始>\n"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        n=num_return_sequences,
        temperature=0.8,
    )

    results = []
    for choice in response.choices:
        text = choice.message.content.strip()
        # 清理序号、空格等
        for line in text.splitlines():
            line = re.sub(r'^(?:\(?\d+\)?[\.、:：]?\s*)', '', line.strip())
            if line and not line.isspace():
                results.append(line)
    return results[:num_return_sequences]

# ========== 文件路径 ==========
input_path = "/home/wangl/sh/11qianwen_filled.jsonl"
output_path = "/home/wangl/sh/50qianwen_filled.jsonl"

# ========== 统计总行数 ==========
with open(input_path, "r", encoding="utf-8") as fp:
    total_lines = sum(1 for _ in fp)

# ========== 断点续跑：读取已处理 ==========
processed = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f_done:
        for ln in f_done:
            try:
                data = json.loads(ln)
                processed.add(data["source"])
            except:
                continue

# ========== 主循环 ==========
with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "a", encoding="utf-8") as fout:

    for idx, line in enumerate(tqdm(fin, total=total_lines, desc="补全改写", unit="条"), start=1):
        data = json.loads(line)
        src = data.get("source", "").strip()

        # 跳过已处理的
        if src in processed:
            continue

        existing = data.get("rewrites", [])
        need = max(0, TARGET_COUNT - len(existing))

        if need > 0:
            new_variants = []
            # 分批调用，避免一次请求太大
            while need > 0:
                cur_batch = min(BATCH_SIZE, need)
                batch_out = rewrite_variants_clean(src, num_return_sequences=cur_batch)
                new_variants.extend(batch_out)
                need -= len(batch_out)

            print(f"\n>>> 样本 {idx} 追加 {len(new_variants)} 条改写 <<<\n原文：{src}\n")
            for i, v in enumerate(new_variants, 1):
                print(f"{i}. {v}")

            existing.extend(new_variants)
            data["rewrites"] = existing

        # 写入文件
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        fout.flush()
        processed.add(src)

print(f"\n✅ 全部处理完成，结果保存在：{output_path}")
