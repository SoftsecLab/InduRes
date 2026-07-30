import argparse
import json
import math
import re
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


PUNCT_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)


def normalize_token(tok: str) -> str:
    tok = str(tok).strip()
    tok = tok.strip("“”‘’\"'（）()[]【】《》<>，。！？；：、,.!?;: \n\t")
    return tok.strip()


def load_spacy():
    import spacy
    print("[INFO] loading spaCy zh_core_web_sm ...")
    nlp = spacy.load("zh_core_web_sm")
    print("[INFO] spaCy loaded.")
    return nlp


def get_first_existing(row, keys, default=""):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def get_label(row):

    if "source_label" in row:
        v = str(row["source_label"]).lower()

        if "human" in v:
            return "human"

        if "machine" in v or "chatgpt" in v or "gpt" in v:
            return "machine"


    if "label" in row:

        v = str(row["label"]).lower()

        if v in ["0", "human"]:
            return "human"

        if v in ["1", "machine"]:
            return "machine"


    return "unknown"



def get_rewrite_text(row):

    return get_first_existing(
        row,
        [
            "rewrite_text",
            "rewrite",
            "rewritten_text",
            "output_text",
            "output",
            "generated_text",
            "generation",
            "response",
            "content",
        ],
        "",
    )



def get_source_text(row):

    return get_first_existing(
        row,
        [
            "source_text",
            "original_text",
            "input_text",
            "prompt_text",
            "text_source",
            "source",
        ],
        "",
    )



def get_source_group_key(row):

    for k in [
        "task_id",
        "source_id",
        "sample_id",
        "item_id",
        "instance_id"
    ]:

        if k in row and row[k] is not None:
            return str(row[k])


    label = get_label(row)

    source_text = str(get_source_text(row))

    question = str(row.get("question", ""))


    return f"{hash(question + source_text)}_{label}"



def get_pair_key(row, source_group_key):

    for k in [
        "pair_id",
        "topic_id",
        "question_id",
        "qid",
        "question"
    ]:

        if k in row and row[k] is not None:
            return str(row[k])


    s = str(source_group_key)

    s = re.sub(
        r"[_\-]?(human|machine|chatgpt|gpt)$",
        "",
        s,
        flags=re.I
    )

    return s



def read_jsonl(path):

    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                rows.append(json.loads(line))


    return rows




class ParserCache:


    def __init__(self, nlp, max_chars=3000):

        self.nlp = nlp

        self.cache = {}

        self.max_chars = max_chars



    def parse(self, text):

        text = str(text or "").strip()


        if text in self.cache:

            return self.cache[text]


        doc = self.nlp(text[:self.max_chars])


        words = []
        deps = []
        positions = []
        heads = []


        for t in doc:

            w = normalize_token(t.text)


            if not w:
                continue


            if t.is_space or t.is_punct or PUNCT_RE.match(w):

                continue



            dep = t.dep_ if t.dep_ else "dep"


            words.append(w)

            deps.append(dep)

            positions.append(t.i)

            heads.append(t.head.i)



        out = {

            "words": words,

            "deps": deps,

            "positions": positions,

            "heads": heads,

            "dep_counter": Counter(deps),

        }


        self.cache[text] = out


        return out




def norm_counter(counter):

    total = sum(counter.values())

    if total <= 0:

        return {}


    return {
        k:v/total
        for k,v in counter.items()
    }




def l1_dist_counter(c1,c2):

    p = norm_counter(c1)

    q = norm_counter(c2)


    keys = set(p)|set(q)


    if not keys:

        return np.nan


    return sum(
        abs(
            p.get(k,0)-q.get(k,0)
        )
        for k in keys
    )



def shannon_entropy_from_counts(counts):

    total = sum(counts.values())

    if total <=0:

        return np.nan


    h=0.0


    for c in counts.values():

        p=c/total

        if p>0:

            h-=p*math.log2(p)


    return h



def align_pairs(src_words,rw_words):

    pairs=[]


    sm=SequenceMatcher(
        a=src_words,
        b=rw_words,
        autojunk=False
    )


    for tag,i1,i2,j1,j2 in sm.get_opcodes():


        if tag=="equal":

            for a,b in zip(
                range(i1,i2),
                range(j1,j2)
            ):

                pairs.append((a,b))


        elif tag=="replace":

            n=min(
                i2-i1,
                j2-j1
            )

            for k in range(n):

                pairs.append(
                    (
                        i1+k,
                        j1+k
                    )
                )


    if not pairs:

        n=min(
            len(src_words),
            len(rw_words)
        )

        pairs=[
            (i,i)
            for i in range(n)
        ]


    return pairs



def transition_counts(src_parse,rw_parse):

    pairs=align_pairs(
        src_parse["words"],
        rw_parse["words"]
    )


    counts=Counter()


    for i,j in pairs:

        if i<len(src_parse["deps"]) and j<len(rw_parse["deps"]):

            counts[
                (
                    src_parse["deps"][i],
                    rw_parse["deps"][j]
                )
            ]+=1


    return counts

def ptrans_features(counts):

    total=sum(counts.values())


    if total<=0:

        return {

            "Ptrans_total_aligned":0,

            "Ptrans_self_rate":np.nan,

            "Ptrans_changed_rate":np.nan,

            "Ptrans_entropy":np.nan,

            "Ptrans_entropy_norm":np.nan,

            "Ptrans_max_prob":np.nan,

            "Ptrans_unique_n":0,

        }



    src_to_tgt=defaultdict(Counter)


    self_count=0


    for (ds,dr),c in counts.items():

        src_to_tgt[ds][dr]+=c

        if ds==dr:

            self_count+=c



    weighted_entropy=0

    weighted_entropy_norm=0

    weighted_max_prob=0



    for ds,tgt_counter in src_to_tgt.items():

        src_total=sum(tgt_counter.values())

        weight=src_total/total


        h=shannon_entropy_from_counts(tgt_counter)


        n_unique=len(tgt_counter)


        h_norm=h/math.log2(n_unique) if n_unique>1 else 0


        max_prob=max(tgt_counter.values())/src_total


        weighted_entropy+=weight*h

        weighted_entropy_norm+=weight*h_norm

        weighted_max_prob+=weight*max_prob



    return {


        "Ptrans_total_aligned":total,

        "Ptrans_self_rate":self_count/total,

        "Ptrans_changed_rate":1-self_count/total,

        "Ptrans_entropy":weighted_entropy,

        "Ptrans_entropy_norm":weighted_entropy_norm,

        "Ptrans_max_prob":weighted_max_prob,

        "Ptrans_unique_n":len(counts),

    }



# ===============================
# Lexical Initiation Variability
# ===============================


def normalized_edit_distance(seq1,seq2):

    seq1=list(seq1)

    seq2=list(seq2)


    n=len(seq1)

    m=len(seq2)


    if n==0 and m==0:

        return 0.0


    if n==0 or m==0:

        return 1.0



    dp=np.zeros(
        (n+1,m+1),
        dtype=int
    )


    dp[:,0]=np.arange(n+1)

    dp[0,:]=np.arange(m+1)



    for i in range(1,n+1):

        for j in range(1,m+1):


            cost=0 if seq1[i-1]==seq2[j-1] else 1


            dp[i,j]=min(

                dp[i-1,j]+1,

                dp[i,j-1]+1,

                dp[i-1,j-1]+cost

            )



    return dp[n,m]/max(n,m)




def lexical_initiation_variability(rewrite_parses,k=8):

    """
    LIV:
    Average pairwise normalized edit distance
    among the first k tokens of rewrites.
    """


    sequences=[]


    for parsed in rewrite_parses:

        words=parsed.get("words",[])

        if words:

            sequences.append(
                words[:k]
            )


    n=len(sequences)


    if n<2:

        return np.nan



    distances=[]


    for i in range(n):

        for j in range(i+1,n):

            distances.append(

                normalized_edit_distance(
                    sequences[i],
                    sequences[j]
                )

            )



    return float(np.mean(distances))






def group_rows_by_source(rows):

    grouped=defaultdict(list)


    for row in rows:


        status=str(
            row.get(
                "status",
                "ok"
            )
        )


        if status not in [
            "ok",
            "partial_ok"
        ]:

            continue



        rewrite=get_rewrite_text(row)

        source=get_source_text(row)



        if not source or not str(source).strip():

            continue


        if not rewrite or not str(rewrite).strip():

            continue



        grouped[
            get_source_group_key(row)
        ].append(row)



    return grouped






def build_reference_dep_counters(rows,parser):

    grouped=group_rows_by_source(rows)


    ref={

        "human":Counter(),

        "machine":Counter(),

    }


    used=0


    for g,rs in grouped.items():


        label=get_label(rs[0])


        if label not in ref:

            continue



        source=get_source_text(rs[0])


        p=parser.parse(source)


        ref[label].update(
            p["dep_counter"]
        )


        used+=1



    return ref,used






def safe_mean(xs):

    xs=[
        x for x in xs
        if not pd.isna(x)
    ]

    return float(np.mean(xs)) if xs else np.nan





def safe_std(xs):

    xs=[
        x for x in xs
        if not pd.isna(x)
    ]

    if len(xs)>1:

        return float(
            np.std(
                xs,
                ddof=1
            )
        )

    return np.nan







def make_features(
    rows,
    ref_rows,
    parser,
    min_rewrites
):


    grouped=group_rows_by_source(rows)


    ref_counters,ref_used=build_reference_dep_counters(
        ref_rows,
        parser
    )


    print(
        f"[INFO] valid source tasks in input: {len(grouped)}"
    )

    print(
        f"[INFO] reference source tasks used: {ref_used}"
    )



    human_ref=ref_counters["human"]

    machine_ref=ref_counters["machine"]



    corpus_Dfreq_source=l1_dist_counter(
        human_ref,
        machine_ref
    )



    feature_rows=[]



    for g,rs in grouped.items():

        label=get_label(rs[0])


        if label not in [
            "human",
            "machine"
        ]:

            continue



        source=get_source_text(rs[0])


        pair_key=get_pair_key(
            rs[0],
            g
        )



        src_parse=parser.parse(source)



        rewrite_parses=[]



        for row in rs:

            rewrite=get_rewrite_text(row)

            if rewrite and str(rewrite).strip():

                rewrite_parses.append(
                    parser.parse(rewrite)
                )



        if len(rewrite_parses)<min_rewrites:

            continue



        # -------- Dfreq --------


        Dfreq_src_to_human_ref=l1_dist_counter(
            src_parse["dep_counter"],
            human_ref
        )


        Dfreq_src_to_machine_ref=l1_dist_counter(
            src_parse["dep_counter"],
            machine_ref
        )


        Dfreq_ref_margin_machine_positive=(

            Dfreq_src_to_human_ref
            -
            Dfreq_src_to_machine_ref

        )



        Dfreq_source_rewrite=[]

        Dfreq_rewrite_ref_margin=[]



        for rp in rewrite_parses:


            d_sr=l1_dist_counter(
                src_parse["dep_counter"],
                rp["dep_counter"]
            )


            d_h=l1_dist_counter(
                rp["dep_counter"],
                human_ref
            )


            d_m=l1_dist_counter(
                rp["dep_counter"],
                machine_ref
            )


            Dfreq_source_rewrite.append(d_sr)

            Dfreq_rewrite_ref_margin.append(
                d_h-d_m
            )



        # -------- Ptrans --------


        all_trans_counts=Counter()


        for rp in rewrite_parses:

            all_trans_counts.update(
                transition_counts(
                    src_parse,
                    rp
                )
            )


        pt=ptrans_features(
            all_trans_counts
        )



        # -------- LIV --------


        LIV_k5=lexical_initiation_variability(
            rewrite_parses,
            k=5
        )


        LIV_k8=lexical_initiation_variability(
            rewrite_parses,
            k=8
        )


        LIV_k10=lexical_initiation_variability(
            rewrite_parses,
            k=10
        )



        row_out={


            "source_group_key":g,

            "pair_key":pair_key,

            "source_label":label,

            "label":0 if label=="human" else 1,


            "n_rewrites":len(rewrite_parses),



            "corpus_Dfreq_human_machine_ref":
                corpus_Dfreq_source,



            "Dfreq_src_to_human_ref":
                Dfreq_src_to_human_ref,


            "Dfreq_src_to_machine_ref":
                Dfreq_src_to_machine_ref,


            "Dfreq_ref_margin_machine_positive":
                Dfreq_ref_margin_machine_positive,



            "Dfreq_source_rewrite_mean":
                safe_mean(Dfreq_source_rewrite),


            "Dfreq_source_rewrite_std":
                safe_std(Dfreq_source_rewrite),


            "Dfreq_rewrite_ref_margin_mean":
                safe_mean(Dfreq_rewrite_ref_margin),


            **pt,



            "LIV_k5":LIV_k5,

            "LIV_k8":LIV_k8,

            "LIV_k10":LIV_k10,

        }


        feature_rows.append(row_out)



    return pd.DataFrame(feature_rows)

def run_tests(feat):

    tests=[]


    metrics=[

        "Dfreq_src_to_human_ref",

        "Dfreq_src_to_machine_ref",

        "Dfreq_ref_margin_machine_positive",

        "Dfreq_source_rewrite_mean",

        "Dfreq_source_rewrite_std",

        "Dfreq_rewrite_ref_margin_mean",


        "Ptrans_self_rate",

        "Ptrans_changed_rate",

        "Ptrans_entropy",

        "Ptrans_entropy_norm",

        "Ptrans_max_prob",

        "Ptrans_unique_n",


        "LIV_k5",

        "LIV_k8",

        "LIV_k10",

    ]



    try:

        from scipy.stats import (
            ttest_ind,
            mannwhitneyu,
            ttest_rel,
            wilcoxon
        )

        from sklearn.metrics import roc_auc_score



        for metric in metrics:


            if metric not in feat.columns:

                continue



            sub=feat[
                feat["source_label"].isin(
                    [
                        "human",
                        "machine"
                    ]
                )
            ].copy()



            sub=sub.dropna(
                subset=[metric]
            )



            if len(sub)<4:

                continue



            human=sub.loc[
                sub["source_label"]=="human",
                metric
            ]


            machine=sub.loc[
                sub["source_label"]=="machine",
                metric
            ]



            if len(human)>1 and len(machine)>1:


                t=ttest_ind(
                    human,
                    machine,
                    equal_var=False
                )


                u=mannwhitneyu(
                    human,
                    machine,
                    alternative="two-sided"
                )


                auc=roc_auc_score(
                    sub["label"].values,
                    sub[metric].values
                )



                tests.append({

                    "metric":metric,

                    "test_type":"independent",

                    "human_mean":human.mean(),

                    "machine_mean":machine.mean(),

                    "human_minus_machine":
                        human.mean()-machine.mean(),

                    "welch_t_p":t.pvalue,

                    "mannwhitney_p":u.pvalue,

                    "roc_auc_machine_positive":auc,

                    "roc_auc_best_direction":
                        max(
                            auc,
                            1-auc
                        ),

                })



            pivot=sub.pivot_table(

                index="pair_key",

                columns="source_label",

                values=metric,

                aggfunc="mean"

            ).dropna()



            if (
                "human" in pivot.columns
                and
                "machine" in pivot.columns
                and
                len(pivot)>1
            ):


                diff=pivot["human"]-pivot["machine"]


                pt=ttest_rel(
                    pivot["human"],
                    pivot["machine"]
                )


                try:

                    pw=wilcoxon(
                        pivot["human"],
                        pivot["machine"]
                    )

                    wilcoxon_p=pw.pvalue


                except:

                    wilcoxon_p=np.nan



                dz=(
                    diff.mean()/diff.std(ddof=1)
                    if diff.std(ddof=1)!=0
                    else np.nan
                )



                tests.append({

                    "metric":metric,

                    "test_type":"paired",

                    "n_pairs":len(pivot),

                    "human_mean":
                        pivot["human"].mean(),

                    "machine_mean":
                        pivot["machine"].mean(),

                    "human_minus_machine":
                        pivot["human"].mean()
                        -
                        pivot["machine"].mean(),

                    "paired_t_p":
                        pt.pvalue,

                    "wilcoxon_p":
                        wilcoxon_p,

                    "cohen_dz_human_minus_machine":
                        dz,

                })


    except Exception as e:

        print(
            f"[WARN] tests skipped: {e}"
        )



    return pd.DataFrame(tests)







def run_model_eval(feat):

    rows=[]



    feature_sets={


        "Dfreq_only":[

            "Dfreq_src_to_human_ref",

            "Dfreq_src_to_machine_ref",

            "Dfreq_ref_margin_machine_positive",

            "Dfreq_source_rewrite_mean",

            "Dfreq_source_rewrite_std",

            "Dfreq_rewrite_ref_margin_mean",

        ],



        "Ptrans_only":[

            "Ptrans_self_rate",

            "Ptrans_changed_rate",

            "Ptrans_entropy",

            "Ptrans_entropy_norm",

            "Ptrans_max_prob",

            "Ptrans_unique_n",

        ],



        "LexicalInitiation_only":[

            "LIV_k5",

            "LIV_k8",

            "LIV_k10",

        ],


    }



    feature_sets["ProbPath_all"]=(

        feature_sets["Dfreq_only"]

        +

        feature_sets["Ptrans_only"]

        +

        feature_sets["LexicalInitiation_only"]

    )



    try:

        from sklearn.pipeline import make_pipeline

        from sklearn.preprocessing import StandardScaler

        from sklearn.linear_model import LogisticRegression

        from sklearn.model_selection import (
            StratifiedKFold,
            cross_val_predict
        )

        from sklearn.metrics import (
            roc_auc_score,
            accuracy_score,
            f1_score,
            precision_score,
            recall_score
        )



        sub=feat[
            feat["source_label"].isin(
                [
                    "human",
                    "machine"
                ]
            )
        ].copy()



        y=sub["label"].values



        min_class=min(
            np.bincount(y)
        )


        n_splits=min(
            5,
            min_class
        )


        if n_splits<2:

            return pd.DataFrame(rows)



        cv=StratifiedKFold(

            n_splits=n_splits,

            shuffle=True,

            random_state=42

        )



        for name,cols in feature_sets.items():


            cols=[
                c for c in cols
                if c in sub.columns
            ]



            X=sub[cols].copy()


            X=X.replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )


            X=X.fillna(
                X.median(
                    numeric_only=True
                )
            )


            X=X.fillna(0)



            clf=make_pipeline(

                StandardScaler(),

                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )

            )



            prob=cross_val_predict(

                clf,

                X,

                y,

                cv=cv,

                method="predict_proba"

            )[:,1]



            pred=(prob>=0.5).astype(int)



            rows.append({

                "feature_set":name,

                "n_features":len(cols),

                "n_samples":len(sub),

                "cv_splits":n_splits,

                "roc_auc":
                    roc_auc_score(
                        y,
                        prob
                    ),

                "accuracy":
                    accuracy_score(
                        y,
                        pred
                    ),

                "precision_weighted":
                    precision_score(
                        y,
                        pred,
                        average="weighted",
                        zero_division=0
                    ),

                "recall_weighted":
                    recall_score(
                        y,
                        pred,
                        average="weighted",
                        zero_division=0
                    ),

                "f1_weighted":
                    f1_score(
                        y,
                        pred,
                        average="weighted",
                        zero_division=0
                    ),

                "features":
                    "|".join(cols),

            })



    except Exception as e:

        print(
            f"[WARN] model eval skipped: {e}"
        )



    return pd.DataFrame(rows)







def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--input_path",
        required=True
    )


    parser.add_argument(
        "--out_prefix",
        required=True
    )


    parser.add_argument(
        "--analysis_dir",
        default="/home/chy/1/data/hc3_processed/analysis"
    )


    parser.add_argument(
        "--ref_path",
        default=None
    )


    parser.add_argument(
        "--min_rewrites",
        type=int,
        default=3
    )



    args=parser.parse_args()



    analysis_dir=Path(
        args.analysis_dir
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True
    )



    rows=read_jsonl(
        args.input_path
    )



    ref_rows=(

        read_jsonl(args.ref_path)

        if args.ref_path

        else rows

    )



    print(
        f"[INFO] input rows: {len(rows)}"
    )

    print(
        f"[INFO] ref rows: {len(ref_rows)}"
    )



    nlp=load_spacy()


    parser_cache=ParserCache(
        nlp
    )



    feat=make_features(

        rows,

        ref_rows,

        parser_cache,

        args.min_rewrites

    )



    out_sample=analysis_dir / (
        args.out_prefix
        +
        "_sample_features.csv"
    )


    feat.to_csv(
        out_sample,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"[SAVE] {out_sample}"
    )

    numeric_feat = feat.select_dtypes(
        include=["number"]
    )

    group_stats = numeric_feat.groupby(
        feat["source_label"]
    ).agg(
        ["count", "mean", "std", "min", "max"]
    )



    out_group=analysis_dir / (
        args.out_prefix
        +
        "_group_stats.csv"
    )


    group_stats.to_csv(
        out_group,
        encoding="utf-8-sig"
    )



    tests=run_tests(feat)



    out_tests=analysis_dir / (
        args.out_prefix
        +
        "_tests.csv"
    )


    tests.to_csv(
        out_tests,
        index=False,
        encoding="utf-8-sig"
    )



    model_eval=run_model_eval(
        feat
    )



    out_model=analysis_dir / (
        args.out_prefix
        +
        "_model_eval.csv"
    )


    model_eval.to_csv(
        out_model,
        index=False,
        encoding="utf-8-sig"
    )



    print("\n[GROUP MEANS]")

    print(
        feat.groupby(
            "source_label"
        )[
            [
                "Dfreq_source_rewrite_mean",

                "Dfreq_rewrite_ref_margin_mean",

                "Ptrans_self_rate",

                "Ptrans_changed_rate",

                "Ptrans_entropy_norm",

                "LIV_k5",

                "LIV_k8",

                "LIV_k10",
            ]
        ].mean()
    )



    print("\n[MODEL EVAL]")

    print(model_eval)




if __name__=="__main__":

    main()