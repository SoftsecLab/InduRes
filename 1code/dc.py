import json
import numpy as np
import spacy
import ot   # pip install POT

# --- 配置 ---
OUTFILE = "/home/wangl/sh/dc_sampled_filled.json"
INPUT_FILE = "/home/chy/1sampled_filled_cleaned.jsonl"
VERBOSE = False   # True 时会在控制台打印每条样本的(6)分解

print("🔄 Loading spaCy dependency parser...")
nlp = spacy.load("zh_core_web_sm")


def get_depth(token):
    depth = 0
    while token.head != token:
        token = token.head
        depth += 1
    return depth


def extract_syntax_tree(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        tokens.append({
            "text": token.text,
            "dep": token.dep_,
            "head": token.head.i,
            "depth": get_depth(token)
        })
    return tokens


def delta_struct(tok_a, tok_b):

    dep_same = 1 if tok_a['dep'] == tok_b['dep'] else 0
    depth_diff = abs(tok_a['depth'] - tok_b['depth']) / max(tok_a['depth'], tok_b['depth'], 1)


    score = 1 - (0.5 * dep_same + 0.5 * (1 - depth_diff))
    return float(score)


def perturbation_mover_struct(a_text, c_text, return_details=False):

    tree_a = extract_syntax_tree(a_text)
    tree_c = extract_syntax_tree(c_text)
    m, n = len(tree_a), len(tree_c)

    # 成本矩阵 C ∈ [0,1]^{m×n}
    C = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            C[i, j] = delta_struct(tree_a[i], tree_c[j])

    # 均匀边际 r, c
    r = np.ones(m, dtype=float) / max(m, 1)
    c = np.ones(n, dtype=float) / max(n, 1)

    # 最优传输计划 P
    P = ot.emd(r, c, C)                # 满足 P1_n=r, P^T1_m=c
    ot_cost = float((P * C).sum())     # Σ P_ij * C_ij, 已在[0,1]

    if not return_details:
        return ot_cost

    # 额外诊断&可视化用的小统计
    row_err = np.abs(P.sum(axis=1) - r).max() if m and n else 0.0
    col_err = np.abs(P.sum(axis=0) - c).max() if m and n else 0.0
    details = {
        "m": int(m),
        "n": int(n),
        "C_min": float(C.min()) if m and n else 0.0,
        "C_max": float(C.max()) if m and n else 0.0,
        "C_mean": float(C.mean()) if m and n else 0.0,
        "P_sum": float(P.sum()),
        "row_marginal_max_abs_err": float(row_err),
        "col_marginal_max_abs_err": float(col_err)
    }
    return ot_cost, details


def avg_mover_score_struct(original, rewrites):

    K = len(rewrites)
    normalizer = 1.0

    per_item = []
    for c in rewrites:
        mover, dbg = perturbation_mover_struct(original, c, return_details=True)
        per_item.append({
            "rewrite": c,
            "perturbation_mover": round(mover, 6),
            "ot_diagnostics": dbg
        })

    sum_pm = sum(x["perturbation_mover"] for x in per_item)
    avg_mover_score = 1.0 - (sum_pm / (K * normalizer))

    breakdown = {
        "K": K,
        "normalizer": normalizer,
        "sum_perturbation_mover": round(sum_pm, 6),
        "avg_mover_score": round(avg_mover_score, 6),
        "per_rewrite": per_item,
        "computation": "AvgMoverScore = 1 - (1/K) * Σ_i PerturbationMover(A, C_i) / 1"
    }

    if VERBOSE:
        print("\n--- Formula (6) breakdown ---")
        print(f"K = {K}, Normalizer = 1")
        for i, x in enumerate(per_item, 1):
            print(f"  i={i}: PerturbationMover = {x['perturbation_mover']}")
        print(f"Σ PerturbationMover = {round(sum_pm, 6)}")
        print(f"AvgMoverScore = {round(avg_mover_score, 6)}")
        print("-----------------------------\n")

    return avg_mover_score, breakdown


def compute_scores():
    results = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]

    for idx, obj in enumerate(data):
        try:
            original = obj["source"]
            label = obj["label"]
            rewrites = obj["rewrites"]
            if not isinstance(rewrites, list) or len(rewrites) == 0:
                continue

            avg_mover, brk = avg_mover_score_struct(original, rewrites)

            results.append({
                "label": label,
                "original": original,
                "avg_mover": round(avg_mover, 6),
                "formula_6_breakdown": brk
            })
            if VERBOSE:
                print(f"✅ {idx + 1}/{len(data)} Processed")
        except Exception as e:
            print(f"❌ Error in entry {idx + 1}: {e}")

    with open(OUTFILE, 'w', encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

    print(f"\n🎉 Done! Results written to: {OUTFILE} (共 {len(results)} 条)")


if __name__ == "__main__":
    compute_scores()
