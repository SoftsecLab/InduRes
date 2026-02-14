import os
import json
import math
import joblib
import spacy
import numpy as np
from collections import Counter
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE   # SMOTE

# ===============================
# 1. 配置路径
# ===============================
SOURCES = [
    ('/home/chy/tiaoshi/merged_human.jsonl',  0),
    ('/home/chy/tiaoshi/11qianwen_filled.jsonl', 1),
    ('/home/chy/tiaoshi/glm4_machine_rewrites.jsonl', 1),
    ('/home/chy/tiaoshi/rewrite_Yi.jsonl', 1),
    ('/home/chy/tiaoshi/wenxin_machine_wenxin_8.jsonl', 1),
]

GLOBALS_PATH = '/home/chy/tiaoshi/syntax_globals.joblib'
MODEL_PATH   = '/home/chy/tiaoshi/mlp_syntax_model_v2.joblib'

# ===============================
# 2. 加载 spaCy 模型
# ===============================
nlp = spacy.load("zh_core_web_sm", disable=["ner", "senter", "attribute_ruler"])

# ===============================
# 3. 加载数据
# ===============================
records = []

print("\n=== 加载 JSONL 数据 ===")
for path, label in SOURCES:
    cnt_lines = 0
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            rec = json.loads(line)
            for txt in rec.get('rewrites', []):
                records.append((txt, label))
                cnt_lines += 1
    print(f"{path} → {cnt_lines} 条样本（label={label}）")

texts, labels = zip(*records)
labels = np.array(labels)

print(f"\nHuman: {np.sum(labels==0)}, Machine: {np.sum(labels==1)}, Total: {len(labels)}")

# ===============================
# 4. 全局统计
# ===============================
human_texts = [t for t, l in records if l == 0]
machine_texts = [t for t, l in records if l == 1]

def dep_freq_counter(texts, desc):
    cnt, tot = Counter(), 0
    for doc in tqdm(nlp.pipe(texts, batch_size=32), total=len(texts), desc=desc):
        for tok in doc:
            cnt[tok.dep_] += 1
            tot += 1
    for d in cnt:
        cnt[d] /= tot
    return cnt

def dep_trans_probs(texts, desc):
    trans, tot = Counter(), Counter()
    for doc in tqdm(nlp.pipe(texts, batch_size=32), total=len(texts), desc=desc):
        for sent in doc.sents:
            deps = [tok.dep_ for tok in sent]
            for a, b in zip(deps, deps[1:]):
                trans[(a, b)] += 1
                tot[a] += 1
    return {k: v / tot[k[0]] for k, v in trans.items()}

hf = dep_freq_counter(human_texts, "Dep-freq human")
mf = dep_freq_counter(machine_texts, "Dep-freq machine")
all_deps = sorted(set(hf) | set(mf))

Ph = dep_trans_probs(human_texts, "Dep-trans human")
Pm = dep_trans_probs(machine_texts, "Dep-trans machine")
all_pairs = sorted(set(Ph) | set(Pm))

# ---- 首词权重：拆成 A/B/C + 自适应组合 w0 ----
first_cnt = Counter()
for doc in tqdm(nlp.pipe(texts, batch_size=32), total=len(texts), desc="First-word count"):
    if doc:
        first_cnt[doc[0].text] += 1

sum_first = sum(first_cnt.values())
p_first = {w: c / sum_first for w, c in first_cnt.items()}

C_cnt = Counter()
for txt in tqdm(texts, desc="Counting C(w)", unit="text"):
    doc = nlp(txt)
    if not doc:
        continue
    w0_tok = doc[0].text
    for sent in doc.sents:
        deps = [tok.dep_ for tok in sent]
        for a, b in zip(deps, deps[1:]):
            C_cnt[w0_tok] += 1

I_raw = {w: -math.log2(p_first[w]) for w in p_first}

sum_p = sum(p_first.values())
sum_C = sum(C_cnt.values())
sum_I = sum(I_raw.values())

# 归一化得到 A/B/C
w_components = {
    w: (
        p_first[w] / sum_p,                          # A: 频率分布
        (C_cnt[w] / sum_C if sum_C > 0 else 0.0),    # B: 结构复杂度
        I_raw[w] / sum_I if sum_I > 0 else 0.0       # C: 信息量
    )
    for w in p_first
}

# 自适应组合：w0 = A + B + C
w_weight = {w: sum(w_components[w]) for w in w_components}

# 保存全局统计
globals_dict = {
    'hf': hf, 'mf': mf,
    'Ph': Ph, 'Pm': Pm,
    'all_deps': all_deps,
    'all_pairs': all_pairs,
    'w_components': w_components,
    'w_weight': w_weight,
}
joblib.dump(globals_dict, GLOBALS_PATH)
print(f"\nSaved globals to {GLOBALS_PATH}")

# ===============================
# 5. 特征提取
# ===============================
def extract_features(text):
    doc = nlp(text)

    # freq
    cnt_f, tot_f = Counter(), 0
    for tok in doc:
        cnt_f[tok.dep_] += 1
        tot_f += 1
    f_text = {d: cnt_f[d] / tot_f if tot_f else 0 for d in all_deps}

    # trans
    trans_txt, tot_t = Counter(), Counter()
    for sent in doc.sents:
        deps = [tok.dep_ for tok in sent]
        for a, b in zip(deps, deps[1:]):
            trans_txt[(a, b)] += 1
            tot_t[a] += 1

    P_text = {
        p: (trans_txt[p] / tot_t[p[0]] if tot_t[p[0]] > 0 else 0)
        for p in all_pairs
    }

    # ---- A/B/C + w0 ----
    if len(doc) > 0:
        A, B, C_val = w_components.get(doc[0].text, (0.0, 0.0, 0.0))
        w0 = w_weight.get(doc[0].text, A + B + C_val)
    else:
        A, B, C_val, w0 = 0.0, 0.0, 0.0, 0.0

    # d_freq / d_trans（保持原公式）
    d_freq = sum(abs(f_text[d] - mf.get(d, 0)) for d in all_deps) \
           - sum(abs(f_text[d] - hf.get(d, 0)) for d in all_deps)

    d_trans = sum(abs(P_text[p] - Pm.get(p, 0)) for p in all_pairs) \
            - sum(abs(P_text[p] - Ph.get(p, 0)) for p in all_pairs)

    # 返回 6 维特征
    return d_freq, d_trans, A, B, C_val, w0

print("\n=== 特征提取 ===")
features = [extract_features(t) for t in tqdm(texts)]
X = np.array(features)
y = labels

# ===============================
# 6. 数据集划分
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

from collections import Counter as C2
print("\nTrain:", C2(y_train))
print("Test:", C2(y_test))

# ===============================
# 7. 标准化（重要）
# ===============================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ===============================
# 8. SMOTE 过采样
# ===============================
print("\n=== 使用 SMOTE 采样 ===")
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)

print("SMOTE 后训练集:", C2(y_train_res))

# ===============================
# 9. MLP 训练（增强版）
# ===============================
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=300,
    momentum=0.0,
    nesterovs_momentum=False,
    random_state=42,
    verbose=False
)

print("\n=== 开始训练 MLP ===")
mlp.fit(X_train_res, y_train_res)

# ===============================
# 10. 评估
# ===============================
y_pred = mlp.predict(X_test_s)
y_proba = mlp.predict_proba(X_test_s)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['human', 'machine']))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===============================
# 11. 保存模型（含 scaler）
# ===============================
joblib.dump((mlp, scaler), MODEL_PATH)
print(f"\n模型已保存到 {MODEL_PATH}")
