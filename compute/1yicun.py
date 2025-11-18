import os
import json
import math
import joblib
import spacy
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# 1. Configuration
SOURCES = [
    ('/root/autodl-tmp/9/data/cleaned-50renlei_rewrite_yi.jsonl',  0),
    ('/root/autodl-tmp/9/data/mucgec_filtered_rewrite_path.jsonl',  0),
    ('/root/autodl-tmp/9/data/11qianwen_filled.jsonl', 1),
    ('/root/autodl-tmp/9/data/glm4_machine_rewrites.jsonl', 1),
    ('/root/autodl-tmp/9/data/rewrite_Yi.jsonl', 1),
    ('/root/autodl-tmp/9/data/wenxin_machine_wenxin_8.jsonl', 1),
]
GLOBALS_PATH = '/root/autodl-tmp/data3/syntax_globals.joblib'
FEATURES_CSV = '/root/autodl-tmp/data3/syntax_features.csv'
MODEL_PATH   = '/root/autodl-tmp/data3/mlp_syntax_model.joblib'
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_EPOCHS     = 200
PROGRESS_FILE = '/root/autodl-tmp/data3/extract_progress.json'

# 2. Load spaCy model
nlp = spacy.load("zh_core_web_sm", disable=["ner", "senter", "attribute_ruler"])

# 3. Load and split human/machine texts (only rewrite parts)
records = []
for path, label in SOURCES:
    for line in tqdm(open(path, 'r', encoding='utf-8'), desc=f"Loading {path}", unit="line"):
        rec = json.loads(line)
        # Only process the 'rewrites' part (i.e., the machine-generated text)
        for txt in rec.get('rewrites', []):
            records.append((txt, label))

texts, labels = zip(*records)
labels = np.array(labels)

human_texts = [t for t, l in records if l == 0]
machine_texts = [t for t, l in records if l == 1]

# 4. Precompute global statistics
def dep_freq_counter(texts, desc):
    cnt, tot = Counter(), 0
    for doc in tqdm(nlp.pipe(texts, batch_size=32), total=len(texts), desc=desc):
        for tok in doc:
            cnt[tok.dep_] += 1
            tot += 1
    for d in cnt:
        cnt[d] /= tot
    return cnt

hf = dep_freq_counter(human_texts, "Dep-freq human")
mf = dep_freq_counter(machine_texts, "Dep-freq machine")
all_deps = sorted(set(hf) | set(mf))

def dep_trans_probs(texts, desc):
    trans, tot = Counter(), Counter()
    for doc in tqdm(nlp.pipe(texts, batch_size=32), total=len(texts), desc=desc):
        for sent in doc.sents:
            deps = [tok.dep_ for tok in sent]
            for a, b in zip(deps, deps[1:]):
                trans[(a, b)] += 1
                tot[a] += 1
    return {k: v / tot[k[0]] for k, v in trans.items()}

Ph = dep_trans_probs(human_texts, "Dep-trans human")
Pm = dep_trans_probs(machine_texts, "Dep-trans machine")
all_pairs = sorted(set(Ph) | set(Pm))

# First-word weight
first_cnt = Counter()
for doc in tqdm(nlp.pipe(texts, batch_size=32), total=len(texts), desc="First-word count"):
    if doc:
        first_cnt[doc[0].text] += 1
sum_first = sum(first_cnt.values())
p_first = {w: c / sum_first for w, c in first_cnt.items()}

C = Counter()
for txt in tqdm(texts, desc="Counting C(w)", unit="text"):
    doc = nlp(txt)
    if not doc: continue
    w0 = doc[0].text
    for sent in doc.sents:
        deps = [tok.dep_ for tok in sent]
        for a, b in zip(deps, deps[1:]):
            C[w0] += 1

I = {w: -math.log2(p_first[w]) for w in p_first}
sum_p = sum(p_first.values())
sum_C = sum(C.values())
sum_I = sum(I.values())
α1, α2, α3 = 1.0, 1.0, 1.0

w_weight = {
    w: α1 * (p_first[w] / sum_p)
      + α2 * (C[w] / sum_C if sum_C > 0 else 0)
      + α3 * (I[w] / sum_I)
    for w in p_first
}

# 4.1 Save globals
globals_dict = {
    'hf': hf, 'mf': mf,
    'Ph': Ph, 'Pm': Pm,
    'all_deps': all_deps,
    'all_pairs': all_pairs,
    'w_weight': w_weight
}
joblib.dump(globals_dict, GLOBALS_PATH)
print(f"Saved globals to {GLOBALS_PATH}")

# 5. Extract features and save to CSV
def extract_features(text):
    doc = nlp(text)
    cnt_f, tot_f = Counter(), 0
    for tok in doc:
        cnt_f[tok.dep_] += 1
        tot_f += 1
    f_text = {d: cnt_f[d] / tot_f if tot_f else 0 for d in all_deps}

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

    w0 = w_weight.get(doc[0].text, 0.0) if len(doc) > 0 else 0.0

    d_freq = sum(abs(f_text[d] - mf.get(d, 0)) for d in all_deps) \
            - sum(abs(f_text[d] - hf.get(d, 0)) for d in all_deps)
    d_trans = sum(abs(P_text[p] - Pm.get(p, 0)) for p in all_pairs) \
            - sum(abs(P_text[p] - Ph.get(p, 0)) for p in all_pairs)

    return d_freq, d_trans, w0

features = []
for text in tqdm(texts, desc="Extracting features", unit="text"):
    features.append(extract_features(text))

df_feat = pd.DataFrame(features, columns=['d_freq', 'd_trans', 'w0'])
df_feat['label'] = labels
df_feat.to_csv(FEATURES_CSV, index=False)
print(f"Saved features to {FEATURES_CSV}")

# 6. 数据集划分
X = df_feat[['d_freq', 'd_trans', 'w0']].values
y = df_feat['label'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# 7. 使用简单的验证，避免交叉验证
mlp = MLPClassifier(random_state=RANDOM_STATE, max_iter=N_EPOCHS)

# 训练模型
mlp.fit(X_train, y_train)

# 8. 使用优化后的模型进行评估
y_pred = mlp.predict(X_test)
y_proba = mlp.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['human', 'machine']))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ROC AUC 计算只在有两个类别时才有效
if len(np.unique(y_test)) > 1:
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 保存优化后的模型
joblib.dump(mlp, MODEL_PATH)
print(f"Saved optimized model to {MODEL_PATH}")
