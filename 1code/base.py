import os
import json
import torch
import warnings
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# ===============================
# 环境变量
# ===============================
os.environ["TRANSFORMERS_CACHE"] = "/home/chy/hf_cache"
os.environ["HF_HOME"] = "/home/chy/hf_home"
os.environ["SAFETENSORS_FAST"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
torch.cuda.empty_cache()

# ===============================
# 路径设置
# ===============================
model_path = "/home/chy/new/QwenQwen1.5-0.5B-Chat"   # 本地模型
data_path = "/home/chy/jiucuoxunlian.json"           # 数据文件
output_dir = "/home/chy/qwen05b_overfit_full"        # 输出目录

# ===============================
# 加载数据（全量）
# ===============================
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    data = [{"input": x["input"], "target": x["target"]}
            for x in raw_data if x.get("input") and x.get("target")]

print(f"✅ 训练数据加载成功，共 {len(data)} 条")

# ===============================
# 加载 tokenizer & 模型
# ===============================
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    local_files_only=True
)

# ===============================
# 数据预处理（拼接 input+target）
# ===============================
def preprocess(example):
    src = f"请纠正这句话：{example['input'].replace('纠错：','')}"
    tgt = example["target"]
    text = src + "\n答案：" + tgt  # 拼接式

    tokenized = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# ✅ 全量数据都作为训练集
train_dataset = Dataset.from_list(data).map(preprocess)

# ===============================
# DataCollator
# ===============================
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ===============================
# 训练参数（过拟合模式）
# ===============================
try:
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=20,
        learning_rate=1e-4,
        weight_decay=0.0,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="epoch",       # 每轮保存一次
        save_total_limit=1,          # 只保留最后一次
        report_to="none"
    )
except TypeError:
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=20,
        learning_rate=1e-4,
        weight_decay=0.0,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none"
    )

# ===============================
# Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ===============================
# 训练
# ===============================
print(f"🚀 开始全量过拟合训练 ({len(data)} 条, 20 轮)")
trainer.train()

# ===============================
# 保存最终模型
# ===============================
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ 模型和 tokenizer 已保存到 {output_dir}")
