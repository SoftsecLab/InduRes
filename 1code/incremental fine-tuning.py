import os
import pandas as pd
import torch
import warnings
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# ===============================
# 环境设置
# ===============================
os.environ["TRANSFORMERS_CACHE"] = "/home/chy/hf_cache"
os.environ["HF_HOME"] = "/home/chy/hf_home"
os.environ["SAFETENSORS_FAST"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
torch.cuda.empty_cache()

# ===============================
# 路径
# ===============================
prev_model_path = "/home/chy/qwen05b_overfit_full"   # 上次训练好的模型
output_dir = "/home/chy/qwen05b_incremental"         # 增量模型保存路径

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
# 读取 TSV 数据
# ===============================
dfs = [pd.read_csv(f, sep="\t") for f in files]
df = pd.concat(dfs, ignore_index=True)

data = [
    {"source": row["source"], "target": row["target"]}
    for _, row in df.iterrows()
    if pd.notna(row["source"]) and pd.notna(row["target"])
]

print(f"✅ 增量训练数据加载成功，共 {len(data)} 条")

# ===============================
# 加载 tokenizer & 模型
# ===============================
tokenizer = AutoTokenizer.from_pretrained(
    prev_model_path,
    trust_remote_code=True,
    local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    prev_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    local_files_only=True
)

# ===============================
# 数据预处理
# ===============================
def preprocess(example):
    src = f"请纠正这句话：{example['source']}"
    tgt = example["target"]
    text = src + "\n答案：" + tgt
    tokenized = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = Dataset.from_list(data).map(
    preprocess,
    remove_columns=["source", "target"]
)

# ===============================
# DataCollator
# ===============================
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ===============================
# 训练参数（steps 级保存）
# ===============================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.0,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="steps",   # 每隔 steps 保存 checkpoint
    save_steps=500,          # 每 500 step 保存一次
    save_total_limit=2,      # 只保留最近 2 个 checkpoint
    report_to="none",
    remove_unused_columns=False
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
# 自动检测 checkpoint 并训练
# ===============================
checkpoint_dir = None
if os.path.isdir(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # 取最后一个 checkpoint
        checkpoint_dir = os.path.join(output_dir, sorted(checkpoints)[-1])

print("🚀 开始/继续增量训练...")
if checkpoint_dir:
    print(f"👉 从 {checkpoint_dir} 恢复训练")
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    print("👉 第一次训练，从头开始")
    trainer.train()

# ===============================
# 保存最终模型
# ===============================
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ 增量模型和 tokenizer 已保存到 {output_dir}")
