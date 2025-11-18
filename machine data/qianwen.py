import requests
import json
from tqdm import tqdm
import time


API_KEY = "sk-fe6089b062d54ff1aa2088935ce63b6c"

# 通义千问改写接口 URL
URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-rewrite/generation"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_rewrite(text):
    """
    调用通义千问改写接口，将传入的 text 按 'instruction' 要求改写后返回一个字符串。
    已经在 payload 中增加了 "model" 参数，避免 BadRequest.EmptyModel 错误。
    """
    payload = {

        "model": "qwen-v1",
        
        "input": {
            "text": text,
            "instruction": "改写为通顺、保留原意"
        }
    }
    try:

        response = requests.post(URL, headers=headers, json=payload, timeout=10, verify=False)
        if response.status_code == 200:
            data = response.json()

            return data.get("output", {}).get("text", "")
        else:

            print(f"[Error {response.status_code}] {response.text}")
            return ""
    except Exception as e:
        print(f"[Exception] 调用接口时发生异常：{e}")
        return ""

if __name__ == "__main__":

    input_path = '/root/autodl-tmp/mucgec_filtered_rewrite_path.jsonl'
    # 输出文件：将每条 source 对应的 8 条改写结果保存为一行 JSON
    output_path = '/root/autodl-tmp/tongyi_rewrites.jsonl'

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        # tqdm 会在终端显示进度条
        for line in tqdm(fin, desc='Processing lines'):
            obj = json.loads(line.strip())
            source = obj.get("source", "").strip()
            if not source:
                continue

            rewrites = []
            # 为同一句 source 连续生成 8 条改写
            for i in range(8):
                rewritten = get_rewrite(source)
                rewrites.append(rewritten)
                # 为了防止频率过快导致限流，可以适当延迟
                time.sleep(0.5)

            out_obj = {
                "source": source,
                "machine_rewrites": rewrites
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()  # 每次写完都刷新到磁盘

    print(f"所有文本改写完成，结果已保存在：{output_path}")
