#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import spacy
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef
)
from tqdm import tqdm

# ———————————— 1. 参数配置 ————————————
GLOBALS_PATH      = '/root/autodl-tmp/data3/syntax_globals.joblib'
FINAL_MODEL_PATH  = '/root/autodl-tmp/data3/syntax_mlp_seed82.joblib'
TEST_CSV          = '/root/autodl-tmp/data3/correct.csv'

SEED              = 82          # 固定随机种子
EPOCHS            = 100         # 总训练轮数
TEST_SIZE         = 0.20        # 测试集比例
VAL_SIZE_IN_TRAIN = 0.20        # 验证集比例

# ———————————— 2. 加载全局统计 ————————————
globals_dict = joblib.load(GLOBALS_PATH)
hf, mf       = globals_dict['hf'], globals_dict['mf']
Ph, Pm       = globals_dict['Ph'], globals_dict['Pm']
all_deps     = globals_dict['all_deps']
all_pairs    = globals_dict['all_pairs']
w_weight     = globals_dict['w_weight']

nlp = spacy.load("zh_core_web_sm", disable=["ner", "senter", "attribute_ruler"])

# ———————————— 3. 加载数据集 ————————————
df = pd.read_csv(TEST_CSV).dropna(subset=['text', 'label'])
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df[df['label'].notna()]
df['label'] = df['label'].astype(int)

texts = df['text'].astype(str).tolist()
y      = df['label'].tolist()

# ———————————— 4. 特征提取函数 ————————————
def extract_features(text):
    doc = nlp(text)

    cnt_f, tot_f = Counter(), 0
    for tok in doc:
        cnt_f[tok.dep_] += 1
        tot_f += 1
    f_text = {d: (cnt_f[d] / tot_f if tot_f else 0) for d in all_deps}
    d_freq = sum(abs(f_text[d] - mf.get(d, 0)) for d in all_deps) \
           - sum(abs(f_text[d] - hf.get(d, 0)) for d in all_deps)

    trans_txt, tot_t = Counter(), Counter()
    for sent in doc.sents:
        deps = [tok.dep_ for tok in sent]
        for a, b in zip(deps, deps[1:]):
            trans_txt[(a, b)] += 1
            tot_t[a] += 1
    P_text = {p: (trans_txt[p] / tot_t[p[0]] if tot_t[p[0]] else 0)
              for p in all_pairs}
    d_trans = sum(abs(P_text[p] - Pm.get(p, 0)) for p in all_pairs) \
            - sum(abs(P_text[p] - Ph.get(p, 0)) for p in all_pairs)

    first = doc[0].text if doc else ''
    w0 = w_weight.get(first, 0.0)

    return [d_freq, d_trans, w0]

# ———————————— 5. 特征提取（带进度条） ————————————
print("⏳ Extracting features...")
X = np.array([extract_features(t) for t in tqdm(texts, desc="Extracting", ncols=80)])
y = np.array(y)

# ———————————— 6. 划分数据集 ————————————
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VAL_SIZE_IN_TRAIN,
    stratify=y_train_full,
    random_state=SEED
)
classes_ = np.unique(y_train)

# ———————————— 7. 创建 MLP（支持增量训练） ————————————
clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='sgd',
    learning_rate_init=0.01,
    max_iter=1,         # 每次只跑 1 epoch
    warm_start=True,    # 允许继续训练
    random_state=SEED
)

# ———————————— 8. 训练 100 轮，进度条 + 每 10 轮评估 ————————————
print("\n===== 开始训练 (seed=82, 共 100 epoch，每 10 epoch 验证) =====")
for epoch in tqdm(range(1, EPOCHS + 1), desc="Training epochs", ncols=80):
    if epoch == 1:
        clf.partial_fit(X_train, y_train, classes=classes_)
    else:
        clf.partial_fit(X_train, y_train)

    # 每 10 epoch 在验证集评估一次
    if epoch % 10 == 0:
        y_val_pred  = clf.predict(X_val)
        acc  = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        f1   = f1_score(y_val, y_val_pred)
        print(f"Epoch {epoch:3d} | Val Acc: {acc:.4f}  Prec: {prec:.4f}  F1: {f1:.4f}")

# ———————————— 9. 最终测试集评估 ————————————
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n===== 测试集最终评估 (训练完 100 epoch) =====")
print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision      : {precision_score(y_test, y_pred):.4f}")
print(f"Recall         : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score       : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC        : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Balanced Acc.  : {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"MCC            : {matthews_corrcoef(y_test, y_pred):.4f}")

# ———————————— 10. 保存模型 ————————————
joblib.dump(clf, FINAL_MODEL_PATH)
print(f"\n✅ Model (seed=82, 100 epochs) saved to {FINAL_MODEL_PATH}")