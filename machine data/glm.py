import os
import json
from zhipuai import ZhipuAI
from tqdm import tqdm


API_KEY = os.getenv("QINGYAN_API_KEY", "").strip()


print("【Debug】使用的 BigModel API_KEY（前20字符）：", API_KEY[:20], "…")
if not API_KEY:
    raise RuntimeError("❗ 错误：请先将你的 BigModel 通用 API Key 写入环境变量 QINGYAN_API_KEY！")

# ——— 2. 初始化 SDK 客户端 —— 使用通用 Key 调用 GLM-4 系列模型 ——
client = ZhipuAI(api_key=API_KEY)

# ——— 3. 文件路径配置 ——
# 输入文件：每行一个 JSON，形如 {"source": "某句话需要改写"}
input_path = "/home/wangl/sh/mucgec_filtered_rewrite_path.jsonl"
# 输出文件：每行一个 JSON，形如 {"source": "...", "machine_rewrites": ["…", "…", ...]}
output_path = "/home/wangl/sh/glm4_machine_rewrites.jsonl"

# ——— 4. 断点续跑逻辑：如果输出文件已存在，就把已处理的 source 读出来，跳过它们 ——
processed = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                processed.add(rec["source"])
            except:
                continue

# ——— 5. 主循环：逐行读取 input_path，然后调用 GLM-4 接口生成 8 条改写 ——
with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "a", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="调用 GLM-4 批量生成改写"):
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)
        source = data.get("source", "").strip()
        if not source or source in processed:
            continue

        print(f"\n📌 原句：{source}")
        rewrites = []

        for i in range(8):
            try:
                # ——— 修改这一行的 model 参数为你账号支持的 GLM-4 系列模型 ——
                #    常见有 "glm-4"、"glm-4-air"、"glm-4-v" 等，你可以到“模型管理”或者“开发文档”里查询确切名称。
                response = client.chat.completions.create(
                    model="glm-4",    # ← 这里改成你实际可用的 GLM-4 系列模型
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一个 AI 改写助手，"
                                "请将用户提供的这句话改写为意思相同但更正式、严谨的表达："
                            )
                        },
                        {
                            "role": "user",
                            "content": source
                        }
                    ],
                    temperature=0.7,
                    max_tokens=256
                )
                # 从返回结果里取出第一条改写
                rewritten = response.choices[0].message.content.strip()
                print(f"  ✍️ 改写{i+1}：{rewritten}")
            except Exception as e:
                # 如果出现 401（身份验证失败）或 403（无权限）或 429（限流）等，会在这里捕获并打印
                err_msg = str(e)
                print(f"  ❌ 第{i+1}条改写失败：{err_msg}")
                rewritten = "【失败】"
            rewrites.append(rewritten)

        # 把这 8 条改写写入输出文件
        out_obj = {
            "source": source,
            "machine_rewrites": rewrites
        }
        fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        fout.flush()

print("\n✅ 全部改写完成，结果已保存在：", output_path)
