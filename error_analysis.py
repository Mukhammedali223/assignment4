"""
Error analysis and sensitivity checks for the Assignment 4 report (Part 4 and Appendix B).

Inputs:
  results/results_<run_id>.csv      answers of the final run
  results/error_labels.csv          manual category of every question standard RAG got wrong
                                    (made from analysis/error_cases.csv by reading the raw log)
Outputs (results/analysis/):
  error_summary.csv      Appendix B: count and share of every error type
  error_by_subset.csv    error types per dataset/group
  retrieval.csv          retrieval hit rate and accuracy given a hit, per subset
  sensitivity.csv        outdated-label check and scorer check
  fig_errors.png         error types for own questions, <24 h vs 2-7 days
  error_analysis_output.txt

Run:
  python error_analysis.py results/results_<run_id>.csv results/error_labels.csv
"""
import csv
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

from analyze import load, question_level, write_csv, SUBSETS, CONDITIONS

DEFINITIONS = {
    "Stale evidence": "Retrieved pages show an older value (previous version, old fact) and the model reports it",
    "Missing evidence (abstention)": "The answer is not in the retrieved text and the model answers NOT FOUND",
    "Evidence ignored": "The answer is in the retrieved text but the model answers NOT FOUND",
    "Misreading of question or context": "Evidence is usable but the model picks the wrong value or misreads the question",
    "Unreliable source": "The wrong answer comes from a low-quality page (video, shop, social media)",
    "Wrong entity": "The answer belongs to a different package or object with a similar name",
    "Outdated reference label": "The FreshQA label (April 2026) is no longer true; checked against the primary source",
    "Disputed reference label": "The reference answer is debatable; the model gives a defensible alternative",
    "Scorer false negative": "The answer is correct but written differently from every accepted alias",
}

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    results_path, labels_path = sys.argv[1], sys.argv[2]
    out_dir = os.path.join(os.path.dirname(results_path) or ".", "analysis")
    os.makedirs(out_dir, exist_ok=True)

    rows = load(results_path)
    qlevel = question_level(rows)
    with open(labels_path, encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f))
    cat_of = {l["id"]: l["manual_category"] for l in labels}

    ids_of = defaultdict(list)
    first = {}
    for r in rows:
        if r["condition"] == "rag" and r["repeat"] == 1:
            ids_of[(r["dataset"], r["group"])].append(r["id"])
            first[r["id"]] = r

    wrong = sorted(i for (i, c), ok in qlevel.items() if c == "rag" and not ok)
    missing = set(wrong) - set(cat_of)
    if missing:
        sys.exit(f"Questions without a manual label: {sorted(missing)}")
    say(f"RAG questions answered wrongly: {len(wrong)}; manually labelled: {len(cat_of)}")

    counts = Counter(cat_of[i] for i in wrong)
    summary = []
    for cat in DEFINITIONS:
        example = next((l for l in labels if l["manual_category"] == cat), None)
        summary.append({"error_type": cat, "definition": DEFINITIONS[cat], "count": counts[cat],
                        "percent_of_errors": round(100 * counts[cat] / len(wrong), 1),
                        "example_id": example["id"] if example else "",
                        "example": (f"truth {example['ground_truth']} / answer {example['rag_answer']}"
                                    if example else "")})
    say("\nERROR TYPES (standard RAG, question level)")
    for s in summary:
        say(f"{s['error_type']:36s} {s['count']:4d}  {s['percent_of_errors']:5.1f}%")

    by_subset = []
    say("\nERROR TYPES PER SUBSET")
    for ds, grp in SUBSETS:
        c = Counter(cat_of[i] for i in wrong if (first[i]["dataset"], first[i]["group"]) == (ds, grp))
        row = {"dataset": ds, "group": grp, "questions": len(ids_of[(ds, grp)]), "errors": sum(c.values())}
        row.update({cat: c[cat] for cat in DEFINITIONS})
        by_subset.append(row)
        say(f"{ds:9s} {grp:15s} errors {sum(c.values()):3d}  " +
            ", ".join(f"{k}: {v}" for k, v in c.most_common()))

    retrieval = []
    say("\nRETRIEVAL HIT RATE (ground truth in retrieved text) AND RAG ACCURACY GIVEN A HIT")
    for ds, grp in SUBSETS:
        ids = ids_of[(ds, grp)]
        hits = [i for i in ids if int(first[i]["truth_in_context"]) == 1]
        ok_hit = sum(qlevel[(i, "rag")] for i in hits)
        retrieval.append({"dataset": ds, "group": grp, "questions": len(ids), "hits": len(hits),
                          "hit_rate": round(len(hits) / len(ids), 3),
                          "rag_correct_given_hit": ok_hit,
                          "rag_acc_given_hit": round(ok_hit / len(hits), 3) if hits else ""})
        say(f"{ds:9s} {grp:15s} hit {len(hits):3d}/{len(ids):3d} = {len(hits) / len(ids):.3f}   "
            f"correct given hit {ok_hit}/{len(hits)}")
    rec = next(r for r in retrieval if r["group"] == "recent_24h")
    old = next(r for r in retrieval if r["group"] == "older_2_7d")
    odds, p = fisher_exact([[rec["hits"], rec["questions"] - rec["hits"]],
                            [old["hits"], old["questions"] - old["hits"]]], alternative="less")
    say(f"Exploratory: hit rate recent vs older, Fisher one-sided p={p:.4f}, odds ratio {odds:.3f}")
    retrieval.append({"dataset": "own", "group": "recent vs older (exploratory Fisher, one-sided)",
                      "questions": "", "hits": "", "hit_rate": f"odds ratio {odds:.3f}",
                      "rag_correct_given_hit": "", "rag_acc_given_hit": f"p={p:.4f}"})

    sens = []
    say("\nSENSITIVITY 1: FreshQA slow-changing with outdated labels excluded (labels never edited)")
    slow = ids_of[("freshqa", "slow-changing")]
    outdated = [i for i in slow if cat_of.get(i) == "Outdated reference label"]
    kept = [i for i in slow if i not in outdated]
    for cond in CONDITIONS:
        a = sum(qlevel[(i, cond)] for i in slow)
        b = sum(qlevel[(i, cond)] for i in kept)
        sens.append({"check": "outdated labels excluded", "subset": "freshqa slow-changing", "condition": cond,
                     "original": f"{a}/{len(slow)} = {a / len(slow):.3f}",
                     "adjusted": f"{b}/{len(kept)} = {b / len(kept):.3f}"})
        say(f"{cond:11s} all labels {a}/{len(slow)} = {a / len(slow):.3f}   "
            f"without {len(outdated)} outdated {b}/{len(kept)} = {b / len(kept):.3f}")

    say("\nSENSITIVITY 2: standard RAG if scorer false negatives were counted as correct")
    for ds, grp in SUBSETS:
        ids = ids_of[(ds, grp)]
        a = sum(qlevel[(i, "rag")] for i in ids)
        fn = sum(1 for i in ids if cat_of.get(i) == "Scorer false negative")
        sens.append({"check": "scorer false negatives counted as correct", "subset": f"{ds} {grp}",
                     "condition": "rag", "original": f"{a}/{len(ids)} = {a / len(ids):.3f}",
                     "adjusted": f"{a + fn}/{len(ids)} = {(a + fn) / len(ids):.3f}"})
        say(f"{ds:9s} {grp:15s} automatic {a}/{len(ids)} = {a / len(ids):.3f}   "
            f"with {fn} false negatives {a + fn}/{len(ids)} = {(a + fn) / len(ids):.3f}")

    write_csv(os.path.join(out_dir, "error_summary.csv"), summary)
    write_csv(os.path.join(out_dir, "error_by_subset.csv"), by_subset)
    write_csv(os.path.join(out_dir, "retrieval.csv"), retrieval)
    write_csv(os.path.join(out_dir, "sensitivity.csv"), sens)

    cats = [c for c in DEFINITIONS if any(r["dataset"] == "own" and r[c] for r in by_subset)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = {"recent_24h": "Changed < 24 h ago", "older_2_7d": "Changed 2-7 days ago"}
    left = {g: 0 for g in names}
    for cat in cats:
        vals = [next(r for r in by_subset if r["group"] == g)[cat] for g in names]
        bars = ax.barh(list(names.values()), vals, left=[left[g] for g in names], label=cat)
        ax.bar_label(bars, labels=[str(v) if v else "" for v in vals], label_type="center", fontsize=8)
        for g, v in zip(names, vals):
            left[g] += v
    ax.set_xlabel("Number of own questions answered wrongly by standard RAG (34 per group)")
    ax.set_title("Why standard RAG failed on own questions")
    ax.set_xlim(0, 34)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_errors.png"), dpi=200)
    plt.close(fig)

    with open(os.path.join(out_dir, "error_analysis_output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say(f"\nSaved to '{out_dir}'")


if __name__ == "__main__":
    main()
