# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# ================== 配置 ==================
FEATURES_CSV = "/root/autodl-tmp/data3/fused_features.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_EPOCHS = 200

# ================== 加载数据 ==================
df = pd.read_csv(FEATURES_CSV)
print("✅ 已加载特征:", df.shape)

# 输入特征和标签
X = df.drop(columns=["label"]).values
y = df["label"].values

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ================== 1. Baseline 拼接特征 ==================
mlp_concat = MLPClassifier(random_state=RANDOM_STATE, max_iter=N_EPOCHS)
mlp_concat.fit(X_train, y_train)
y_pred_concat = mlp_concat.predict(X_test)
y_proba_concat = mlp_concat.predict_proba(X_test)[:, 1]

print("\n=== Baseline: 拼接特征 ===")
print(classification_report(y_test, y_pred_concat, target_names=['human', 'machine']))
print("Accuracy:", accuracy_score(y_test, y_pred_concat))
print("ROC AUC:", roc_auc_score(y_test, y_proba_concat))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_concat))

# ================== 2. Late Fusion 概率加权 ==================
# 这里简单模拟：将特征分成两部分（前 3 个表面特征 + 后 4 个句法特征）
X1_train, X1_test = X_train[:, :3], X_test[:, :3]  # 表面特征
X2_train, X2_test = X_train[:, 3:], X_test[:, 3:]  # 句法特征

mlp1 = MLPClassifier(random_state=RANDOM_STATE, max_iter=N_EPOCHS)
mlp2 = MLPClassifier(random_state=RANDOM_STATE, max_iter=N_EPOCHS)

mlp1.fit(X1_train, y_train)
mlp2.fit(X2_train, y_train)

proba1 = mlp1.predict_proba(X1_test)[:, 1]
proba2 = mlp2.predict_proba(X2_test)[:, 1]

alpha, beta = 0.3, 0.7
late_proba = alpha * proba1 + beta * proba2
late_pred = (late_proba > 0.5).astype(int)

print("\n=== Late Fusion: 概率加权 (α=0.3, β=0.7) ===")
print(classification_report(y_test, late_pred, target_names=['human', 'machine']))
print("Accuracy:", accuracy_score(y_test, late_pred))
print("ROC AUC:", roc_auc_score(y_test, late_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, late_pred))

# ================== 3. Stacking 融合 ==================
# 第 1 层：用 mlp1 + mlp2 作为基学习器
stack_train = np.vstack([
    mlp1.predict_proba(X1_train)[:, 1],
    mlp2.predict_proba(X2_train)[:, 1]
]).T
stack_test = np.vstack([proba1, proba2]).T

# 第 2 层：用逻辑回归做元学习器
meta = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
meta.fit(stack_train, y_train)

stack_pred = meta.predict(stack_test)
stack_proba = meta.predict_proba(stack_test)[:, 1]

print("\n=== Stacking Fusion ===")
print(classification_report(y_test, stack_pred, target_names=['human', 'machine']))
print("Accuracy:", accuracy_score(y_test, stack_pred))
print("ROC AUC:", roc_auc_score(y_test, stack_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, stack_pred))
