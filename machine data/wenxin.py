import json
import requests
from tqdm import tqdm
import os

# ========= 你的 API 凭证 ==========
API_KEY = "eGpJYr72LGEiMIxAAxtDhNZr"
SECRET_KEY = "EDLSHT9e4mS8JYducZu7aJ5u1n467N3n"

# ========= 获取 access_token ==========
def get_access_token(api_key, secret_key):
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    res = requests.post(url, params=params)
    return res.json()["access_token"]

# ========= 调用文心一言接口生成改写 ==========
def call_ernie(prompt, access_token):
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={access_token}"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "top_p": 0.8
    }
    res = requests.post(url, headers=headers, json=data)
    result = res.json()
    return result.get("result", "").strip()

# ========= 主程序 ==========
input_path = "/home/wangl/sh/mucgec_filtered_rewrite_path.jsonl"  # 输入文件（含source字段）
output_path = "/home/wangl/sh/wenxin_machine_wenxin_8.jsonl"       # 输出文件（追加写入）

access_token = get_access_token(API_KEY, SECRET_KEY)

# 已处理集合（用于断点续跑）
processed = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data["source"])
            except:
                continue

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "a", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="生成机器风格改写"):
        item = json.loads(line)
        source = item["source"].strip()

        if source in processed:
            continue

        print(f"\n📌 原句：{source}")
        rewrites = []

        for i in range(8):
            prompt = f"请将下面这句话改写成意思相同但更正式、严谨的另一种表达方式：\n原句：{source}\n改写："
            try:
                rewrite = call_ernie(prompt, access_token)
                print(f"  ✍️ 改写{i+1}：{rewrite}")
                rewrites.append(rewrite)
            except Exception as e:
                print(f"❌ 第{i+1}条改写失败：{e}")
                rewrites.append("【失败】")

        fout.write(json.dumps({
            "source": source,
            "machine_rewrites": rewrites
        }, ensure_ascii=False) + "\n")
        fout.flush()  # 实时写入
