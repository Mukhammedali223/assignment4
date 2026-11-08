"""
Builds every table, statistic and figure of the Assignment 4 report from ONE results file.
Every number in the report can be found in the files this script writes.

  main_table.csv     accuracy per dataset x group x condition:
                     mean +/- SD over the 3 repeats (strict, normalized = PRIMARY, F1)
                     and 95% bootstrap CI over questions
  stats.csv          H1 (Fisher exact test) and paired condition comparisons
                     (McNemar exact test) with Holm correction, differences and effect sizes
  efficiency.csv     latency (median, IQR, p95), tokens and cost per answer per condition
  error_cases.csv    every question RAG got wrong, for manual error categorisation
  fig_accuracy.png   accuracy by dataset and condition with 95% CI
  fig_recency.png    own questions: accuracy <24 h vs 2-7 days per condition
  fig_latency.png    latency distribution per condition
  analysis_output.txt  everything printed on screen

Question-level result = correct in at least 2 of the 3 repeats (majority vote).

Run:
  python analyze.py results/results_<run_id>.csv
"""
import csv
import os
import random
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, fisher_exact

PRIMARY = "correct_norm"
METRICS = {"strict": "correct_strict", "norm": "correct_norm", "f1": "correct_f1"}
CONDITIONS = ["baseline", "rag", "freshprompt"]
LABELS = {"baseline": "No retrieval", "rag": "Standard RAG", "freshprompt": "FreshPrompt"}
SUBSETS = [("own", "recent_24h"), ("own", "older_2_7d"),
           ("freshqa", "never-changing"), ("freshqa", "slow-changing"), ("simpleqa", "static")]
PRICE_IN, PRICE_OUT = 0.12, 0.48
N_BOOT = 2000
SEED = 42

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["repeat"] = int(r["repeat"])
        for m in METRICS.values():
            r[m] = int(r[m])
    return rows


def question_level(rows, metric=PRIMARY):
    """{(id, condition): 1/0} - majority of the repeats."""
    votes = defaultdict(list)
    for r in rows:
        votes[(r["id"], r["condition"])].append(r[metric])
    return {k: int(sum(v) * 2 > len(v)) for k, v in votes.items()}


def bootstrap_ci(values, rng):
    if not values:
        return (float("nan"), float("nan"))
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(N_BOOT))
    return means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1]


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main_table(rows, ids_of, rng):
    out = []
    qlevel = question_level(rows)
    for ds, grp in SUBSETS:
        ids = ids_of[(ds, grp)]
        if not ids:
            continue
        for cond in CONDITIONS:
            sub = [r for r in rows if r["dataset"] == ds and r["group"] == grp and r["condition"] == cond]
            if not sub:
                continue
            row = {"dataset": ds, "group": grp, "condition": cond, "n_questions": len(ids)}
            for name, m in METRICS.items():
                per_repeat = []
                for rep in sorted({r["repeat"] for r in sub}):
                    vals = [r[m] for r in sub if r["repeat"] == rep]
                    per_repeat.append(sum(vals) / len(vals))
                row[f"{name}_mean"] = round(statistics.mean(per_repeat), 3)
                row[f"{name}_sd"] = round(statistics.stdev(per_repeat), 3) if len(per_repeat) > 1 else 0.0
            q = [qlevel[(i, cond)] for i in ids if (i, cond) in qlevel]
            lo, hi = bootstrap_ci(q, rng)
            row["question_level_acc"] = round(sum(q) / len(q), 3)
            row["ci95_low"], row["ci95_high"] = round(lo, 3), round(hi, 3)
            out.append(row)
    return out


def fisher_h1(qlevel, ids_of, cond):
    recent = [qlevel[(i, cond)] for i in ids_of[("own", "recent_24h")] if (i, cond) in qlevel]
    older = [qlevel[(i, cond)] for i in ids_of[("own", "older_2_7d")] if (i, cond) in qlevel]
    a, b = sum(recent), len(recent) - sum(recent)
    c, d = sum(older), len(older) - sum(older)
    odds, p = fisher_exact([[a, b], [c, d]], alternative="less")
    return {"test": "Fisher exact, one-sided (recent < older)",
            "comparison": f"{cond}: own recent_24h vs older_2_7d",
            "a": f"{a}/{len(recent)}", "b": f"{c}/{len(older)}",
            "acc_a": round(a / len(recent), 3), "acc_b": round(c / len(older), 3),
            "difference_pp": round(100 * (a / len(recent) - c / len(older)), 1),
            "effect_size": f"odds ratio {odds:.3f}", "statistic": "", "p": p}


def mcnemar(qlevel, ids, cond_a, cond_b, label):
    pairs = [(qlevel[(i, cond_a)], qlevel[(i, cond_b)]) for i in ids
             if (i, cond_a) in qlevel and (i, cond_b) in qlevel]
    only_a = sum(1 for x, y in pairs if x == 1 and y == 0)
    only_b = sum(1 for x, y in pairs if x == 0 and y == 1)
    n = len(pairs)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    acc_a = sum(x for x, _ in pairs) / n
    acc_b = sum(y for _, y in pairs) / n
    return {"test": "McNemar exact, two-sided", "comparison": f"{label}: {cond_a} vs {cond_b}",
            "a": f"{sum(x for x, _ in pairs)}/{n}", "b": f"{sum(y for _, y in pairs)}/{n}",
            "acc_a": round(acc_a, 3), "acc_b": round(acc_b, 3),
            "difference_pp": round(100 * (acc_a - acc_b), 1),
            "effect_size": f"discordant {only_a} vs {only_b}", "statistic": only_a, "p": p}


def stats_table(rows, ids_of):
    qlevel = question_level(rows)
    primary = fisher_h1(qlevel, ids_of, "rag")
    primary["role"] = "H1 primary"
    secondary = [fisher_h1(qlevel, ids_of, "freshprompt")]
    all_ids = {name: [i for key, ids in ids_of.items() if key[0] == name for i in ids]
               for name in ["own", "freshqa", "simpleqa"]}
    for name, ids in all_ids.items():
        secondary.append(mcnemar(qlevel, ids, "rag", "baseline", name))
        secondary.append(mcnemar(qlevel, ids, "freshprompt", "rag", name))
    for s, p_adj in zip(secondary, holm([s["p"] for s in secondary])):
        s["role"] = "secondary"
        s["p_holm"] = p_adj
    primary["p_holm"] = primary["p"]
    out = [primary] + secondary
    for s in out:
        s["p"], s["p_holm"] = round(s["p"], 4), round(s["p_holm"], 4)
    return out


def efficiency(rows):
    out = []
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond and r.get("latency_ms")]
        if not sub:
            continue
        lat = sorted(float(r["latency_ms"]) for r in sub)
        q = statistics.quantiles(lat, n=20)
        tin = sum(int(r["prompt_tokens"] or 0) for r in sub)
        tout = sum(int(r["completion_tokens"] or 0) for r in sub)
        out.append({"condition": cond, "answers": len(sub),
                    "latency_median_ms": round(statistics.median(lat)),
                    "latency_p25_ms": round(q[4]), "latency_p75_ms": round(q[14]),
                    "latency_p95_ms": round(q[18]),
                    "avg_prompt_tokens": round(tin / len(sub)),
                    "avg_completion_tokens": round(tout / len(sub), 1),
                    "cost_per_1000_answers_usd": round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6 / len(sub) * 1000, 4)})
    search = sorted({(r["id"], float(r["search_latency_ms"])) for r in rows if r.get("search_latency_ms")})
    if search:
        vals = [v for _, v in search]
        out.append({"condition": "search (Tavily, once per question)", "answers": len(vals),
                    "latency_median_ms": round(statistics.median(vals)),
                    "latency_p25_ms": round(statistics.quantiles(vals, n=4)[0]),
                    "latency_p75_ms": round(statistics.quantiles(vals, n=4)[2]),
                    "latency_p95_ms": round(statistics.quantiles(vals, n=20)[18]),
                    "avg_prompt_tokens": "", "avg_completion_tokens": "", "cost_per_1000_answers_usd": ""})
    return out


def error_cases(rows):
    qlevel = question_level(rows)
    first = {(r["id"], r["condition"]): r for r in rows if r["repeat"] == 1}
    out = []
    for (qid, cond), ok in sorted(qlevel.items()):
        if cond != "rag" or ok:
            continue
        r = first[(qid, cond)]
        out.append({"id": qid, "dataset": r["dataset"], "group": r["group"],
                    "ground_truth": r["ground_truth"], "rag_answer": r["answer"],
                    "truth_in_context": r["truth_in_context"], "auto_error_type": r["error_type"],
                    "manual_category": "", "comment": ""})
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fig_accuracy(table, path):
    names = [f"{d}\n{g}" for d, g in SUBSETS if any(t["dataset"] == d and t["group"] == g for t in table)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.26
    for k, cond in enumerate(CONDITIONS):
        vals, err_lo, err_hi = [], [], []
        for d, g in SUBSETS:
            t = next((t for t in table if t["dataset"] == d and t["group"] == g and t["condition"] == cond), None)
            if t is None:
                continue
            vals.append(t["question_level_acc"])
            err_lo.append(t["question_level_acc"] - t["ci95_low"])
            err_hi.append(t["ci95_high"] - t["question_level_acc"])
        xs = [i + (k - 1) * width for i in range(len(vals))]
        ax.bar(xs, vals, width, yerr=[err_lo, err_hi], capsize=3, label=LABELS[cond])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Accuracy (share of questions correct)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by dataset and condition (error bars: 95% bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_recency(table, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.36
    for k, (grp, label) in enumerate([("recent_24h", "Changed < 24 h ago"), ("older_2_7d", "Changed 2-7 days ago")]):
        vals, lo, hi = [], [], []
        for cond in CONDITIONS:
            t = next(t for t in table if t["dataset"] == "own" and t["group"] == grp and t["condition"] == cond)
            vals.append(t["question_level_acc"])
            lo.append(t["question_level_acc"] - t["ci95_low"])
            hi.append(t["ci95_high"] - t["question_level_acc"])
        xs = [i + (k - 0.5) * width for i in range(len(CONDITIONS))]
        bars = ax.bar(xs, vals, width, yerr=[lo, hi], capsize=3, label=label)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=3, fontsize=8)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Accuracy (share of questions correct)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Own questions: accuracy by information recency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_latency(rows, path):
    data = [[float(r["latency_ms"]) / 1000 for r in rows if r["condition"] == c and r.get("latency_ms")]
            for c in CONDITIONS]
    if not all(data):
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(CONDITIONS) + 1))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Model response time, seconds")
    ax.set_title("Latency per answer (box: IQR, line: median, outliers hidden)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1]
    out_dir = os.path.join(os.path.dirname(path) or ".", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    rows = load(path)
    rng = random.Random(SEED)

    ids_of = defaultdict(list)
    for r in rows:
        if r["condition"] == "rag" and r["repeat"] == 1:
            ids_of[(r["dataset"], r["group"])].append(r["id"])
    out_window = sorted({r["id"] for r in rows if r["in_window"] == "NO"})
    say(f"Results file: {path}")
    say(f"Answers: {len(rows)}; questions: {sum(len(v) for v in ids_of.values())}")
    say(f"Questions outside their time window: {out_window or 'none'}")

    table = main_table(rows, ids_of, rng)
    stats = stats_table(rows, ids_of)
    eff = efficiency(rows)
    errors = error_cases(rows)

    say("\nMAIN TABLE (mean +/- SD over repeats; question-level accuracy with 95% CI)")
    for t in table:
        say(f"{t['dataset']:9s} {t['group']:15s} {t['condition']:11s} n={t['n_questions']:3d}  "
            f"norm {t['norm_mean']:.3f}+/-{t['norm_sd']:.3f}  strict {t['strict_mean']:.3f}  "
            f"f1 {t['f1_mean']:.3f}  q-level {t['question_level_acc']:.3f} "
            f"[{t['ci95_low']:.3f}, {t['ci95_high']:.3f}]")
    say("\nSTATISTICAL TESTS")
    for s in stats:
        say(f"{s['role']:10s} {s['comparison']:42s} {s['a']:>7s} vs {s['b']:>7s}  "
            f"diff {s['difference_pp']:+.1f} pp  {s['effect_size']:24s} p={s['p']:.4f}  "
            f"p_holm={s['p_holm']:.4f}  ({s['test']})")
    say("\nEFFICIENCY")
    for e in eff:
        say(e)
    say(f"\nRAG question-level errors: {len(errors)}")
    by_type = defaultdict(int)
    for e in errors:
        by_type[f"{e['dataset']}:{e['auto_error_type']}"] += 1
    say(dict(by_type))

    write_csv(os.path.join(out_dir, "main_table.csv"), table)
    write_csv(os.path.join(out_dir, "stats.csv"), stats)
    write_csv(os.path.join(out_dir, "efficiency.csv"), eff)
    write_csv(os.path.join(out_dir, "error_cases.csv"), errors)
    fig_accuracy(table, os.path.join(out_dir, "fig_accuracy.png"))
    fig_recency(table, os.path.join(out_dir, "fig_recency.png"))
    fig_latency(rows, os.path.join(out_dir, "fig_latency.png"))
    with open(os.path.join(out_dir, "analysis_output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say(f"\nSaved to '{out_dir}'")


if __name__ == "__main__":
    main()
