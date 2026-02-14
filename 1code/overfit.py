import os
import json
import torch
import warnings
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from datasets import Dataset

# ✅ 环境变量
os.environ["TRANSFORMERS_CACHE"] = "/home/chy/hf_cache"
os.environ["HF_HOME"] = "/home/chy/hf_home"
os.environ["SAFETENSORS_FAST"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
torch.cuda.empty_cache()

# ✅ 路径
model_path = "/home/chy/new/QwenQwen1.5-0.5B-Chat"
data_path = "/home/chy/new/all_correction.json"
output_dir = "/home/chy/qwen05b_correction_output"
log_dir = "/home/chy/qwen05b_logs"

# ✅ 加载数据
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    data = [{"input": x["input"], "target": x["target"]}
            for x in raw_data if x.get("input") and x.get("target")]
print(f"✅ 数据加载成功，共 {len(data)} 条")

# ✅ 数据扩充（让模型先“记住”）
repeat_factor = 10
data = data * repeat_factor
print(f"✅ 数据扩充后，共 {len(data)} 条")

# ✅ 加载 tokenizer & 模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

# ✅ 数据预处理
def preprocess(example):
    src_text = f"请纠正这句话：{example['input'].replace('纠错：','')}"
    tgt_text = example["target"]

    model_inputs = tokenizer(
        src_text,
        max_length=256,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        tgt_text,
        max_length=256,
        padding="max_length",
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

split_idx = int(0.9 * len(data))
train_dataset = Dataset.from_list(data[:split_idx]).map(preprocess)
eval_dataset = Dataset.from_list(data[split_idx:]).map(preprocess)

# ✅ DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# ✅ 兼容 TrainingArguments (evaluation_strategy / eval_strategy)
def get_training_args():
    try:
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir=log_dir,
            logging_steps=50,
            evaluation_strategy="steps",   # 新版
            eval_steps=200,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            report_to="none"
        )
    except TypeError:
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir=log_dir,
            logging_steps=50,
            eval_strategy="steps",   # 旧版
            eval_steps=200,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            report_to="none"
        )
    return args

training_args = get_training_args()

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ✅ 自动续训
checkpoint_dir = None
if os.path.isdir(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        checkpoint_dir = os.path.join(output_dir, checkpoints[-1])

if checkpoint_dir:
    print(f"🔄 从 {checkpoint_dir} 恢复训练")
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    print("🚀 从头开始训练")
    trainer.train()

# ✅ 保存最终模型
final_path = os.path.join(output_dir, "final_model")
trainer.save_model(final_path)
print(f"✅ 模型已保存到：{final_path}")
