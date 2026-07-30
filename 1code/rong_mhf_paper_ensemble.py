#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PaperMHF three-path fusion with validation-selected multi-seed models.

合法提升方式：
1. threshold 从 train700 grouped OOF 中选择；
2. 多 seed final model 只按 validation AUC 选择或 top-k ensemble；
3. test200 只做最终报告，不用于选模型。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from rong_mhf_paper import (
    DEFAULT_BASE_TRAIN_CSV,
    DEFAULT_BASE_TEST_CSV,
    DEFAULT_PERTURB_TRAIN_CSV,
    DEFAULT_PERTURB_TEST_CSV,
    load_base_data,
    load_perturbation_data,
    merge_base_and_perturbation,
    prepare_group_column,
    validate_feature_columns,
    grouped_oof_predictions,
    split_train_val_by_group,
    train_one_mhf,
    evaluate_predictions,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_train_csv", default=DEFAULT_BASE_TRAIN_CSV)
    parser.add_argument("--base_test_csv", default=DEFAULT_BASE_TEST_CSV)
    parser.add_argument("--perturb_train_csv", default=DEFAULT_PERTURB_TRAIN_CSV)
    parser.add_argument("--perturb_test_csv", default=DEFAULT_PERTURB_TEST_CSV)

    parser.add_argument(
        "--out_dir",
        default="/home/chy/1/data/hc3_processed/analysis/mhf_paper_three_path_fusion_ensemble",
    )

    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--heads", type=int, default=4)

    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.15)

    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46,47,48,49,50,51",
        help="final models seeds",
    )

    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def find_best_threshold(y_true, prob, metric):
    best_t = 0.5
    best_score = -1.0

    for t in np.linspace(0.2, 0.8, 121):
        pred = (prob >= t).astype(int)

        if metric == "accuracy":
            score = accuracy_score(y_true, pred)
        elif metric == "weighted_f1":
            score = f1_score(y_true, pred, average="weighted")
        else:
            raise ValueError(metric)

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score


def simple_eval(y, p, threshold):
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "machine_f1": f1_score(y, pred, pos_label=1),
        "roc_auc": roc_auc_score(y, p),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "pred": pred,
    }


def main():
    args = parse_args()
    set_seed(args.random_state)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[LOAD DATA]")

    base_train = load_base_data(args.base_train_csv, "train700")
    base_test = load_base_data(args.base_test_csv, "test200")

    perturb_train = load_perturbation_data(args.perturb_train_csv, "train700")
    perturb_test = load_perturbation_data(args.perturb_test_csv, "test200")

    train_df = merge_base_and_perturbation(base_train, perturb_train, "train700")
    test_df = merge_base_and_perturbation(base_test, perturb_test, "test200")

    train_df = prepare_group_column(train_df, "train700")
    test_df = prepare_group_column(test_df, "test200")

    validate_feature_columns(train_df, test_df)

    y_train = train_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    print("=" * 80)
    print("[STEP 1] grouped OOF thresholds on train700")

    oof_prob, oof_gate, oof_auc, fold_aucs = grouped_oof_predictions(train_df, args)

    t_f1, oof_best_f1 = find_best_threshold(y_train, oof_prob, metric="weighted_f1")
    t_acc, oof_best_acc = find_best_threshold(y_train, oof_prob, metric="accuracy")

    print("=" * 80)
    print("Grouped OOF AUC:", round(oof_auc, 5))
    print("Fold AUCs:", [round(x, 5) for x in fold_aucs])
    print("OOF best threshold by weighted F1:", round(t_f1, 4), "score:", round(oof_best_f1, 5))
    print("OOF best threshold by accuracy:", round(t_acc, 4), "score:", round(oof_best_acc, 5))
    print("OOF average gate:", np.round(oof_gate.mean(axis=0), 4))

    print("=" * 80)
    print("[STEP 2] multi-seed final PaperMHF models")

    final_train, final_val = split_train_val_by_group(
        train_df,
        args.cv_splits,
        args.random_state,
    )

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    seed_payloads = []

    for seed in seeds:
        print("-" * 80)
        print("[FINAL SEED]", seed)

        payload = train_one_mhf(
            train_df=final_train,
            val_df=final_val,
            test_df=test_df,
            args=args,
            seed=seed,
        )

        val_auc = payload["best_val_auc"]
        p_test = payload["test_prob"]
        gate_test = payload["test_gate"]

        m05 = simple_eval(y_test, p_test, 0.5)

        print(
            f"seed={seed} | val_auc={val_auc:.5f} | "
            f"test_auc={m05['roc_auc']:.5f} | "
            f"test_acc@0.5={m05['accuracy']:.5f} | "
            f"test_f1@0.5={m05['weighted_f1']:.5f} | "
            f"gate={np.round(gate_test.mean(axis=0), 4)}"
        )

        seed_payloads.append({
            "seed": seed,
            "val_auc": val_auc,
            "test_prob": p_test,
            "test_gate": gate_test,
            "best_epoch": payload["best_epoch"],
        })

    seed_payloads = sorted(
        seed_payloads,
        key=lambda x: x["val_auc"],
        reverse=True,
    )

    rows = []
    prediction_tables = []

    thresholds = [
        ("0.5", 0.5),
        ("oof_weighted_f1", t_f1),
        ("oof_accuracy", t_acc),
    ]

    # 单 seed：只按 validation AUC 选，不看 test 选
    best_seed_payload = seed_payloads[0]

    # top-k ensemble：也只按 validation AUC 选 top-k
    top_k = min(args.top_k, len(seed_payloads))
    top_payloads = seed_payloads[:top_k]

    candidates = []

    candidates.append({
        "selection": "best_single_by_val_auc",
        "seeds": str([best_seed_payload["seed"]]),
        "mean_val_auc": best_seed_payload["val_auc"],
        "prob": best_seed_payload["test_prob"],
        "gate": best_seed_payload["test_gate"],
    })

    candidates.append({
        "selection": f"top_{top_k}_ensemble_by_val_auc",
        "seeds": str([x["seed"] for x in top_payloads]),
        "mean_val_auc": float(np.mean([x["val_auc"] for x in top_payloads])),
        "prob": np.mean([x["test_prob"] for x in top_payloads], axis=0),
        "gate": np.mean([x["test_gate"] for x in top_payloads], axis=0),
    })

    candidates.append({
        "selection": "all_seed_ensemble",
        "seeds": str([x["seed"] for x in seed_payloads]),
        "mean_val_auc": float(np.mean([x["val_auc"] for x in seed_payloads])),
        "prob": np.mean([x["test_prob"] for x in seed_payloads], axis=0),
        "gate": np.mean([x["test_gate"] for x in seed_payloads], axis=0),
    })

    for cand in candidates:
        p = cand["prob"]
        gate = cand["gate"]

        for threshold_type, threshold in thresholds:
            metrics = simple_eval(y_test, p, threshold)

            rows.append({
                "model": "PaperMHF_three_path",
                "selection": cand["selection"],
                "seeds": cand["seeds"],
                "threshold_type": threshold_type,
                "threshold": threshold,
                "mean_val_auc": cand["mean_val_auc"],
                "grouped_oof_auc": oof_auc,
                "oof_best_weighted_f1": oof_best_f1,
                "oof_best_accuracy": oof_best_acc,
                "gate_structural": float(gate[:, 0].mean()),
                "gate_revision": float(gate[:, 1].mean()),
                "gate_prob": float(gate[:, 2].mean()),
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "machine_f1": metrics["machine_f1"],
                "roc_auc": metrics["roc_auc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            })

        pred_cols = [
            c for c in ["task_id", "sample_id", "topic_id", "source_label", "label"]
            if c in test_df.columns
        ]

        pred_df = test_df[pred_cols].copy()
        pred_df["selection"] = cand["selection"]
        pred_df["seeds"] = cand["seeds"]
        pred_df["prob_machine"] = p
        pred_df["gate_structural"] = gate[:, 0]
        pred_df["gate_revision"] = gate[:, 1]
        pred_df["gate_prob"] = gate[:, 2]
        prediction_tables.append(pred_df)

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["accuracy", "weighted_f1", "roc_auc"],
        ascending=False,
    )

    out_metrics = out_dir / "paper_mhf_three_path_ensemble_metrics.csv"
    out_predictions = out_dir / "paper_mhf_three_path_ensemble_predictions.csv"

    result.to_csv(out_metrics, index=False, encoding="utf-8-sig")
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        out_predictions,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("[BEST BY ACCURACY]")
    print(
        result[[
            "selection",
            "seeds",
            "threshold_type",
            "threshold",
            "accuracy",
            "weighted_f1",
            "machine_f1",
            "roc_auc",
            "mean_val_auc",
            "gate_structural",
            "gate_revision",
            "gate_prob",
            "tn",
            "fp",
            "fn",
            "tp",
        ]].to_string(index=False)
    )

    print()
    print("Saved metrics to:", out_metrics)
    print("Saved predictions to:", out_predictions)


if __name__ == "__main__":
    main()
