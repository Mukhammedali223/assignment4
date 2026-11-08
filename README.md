# Information recency and RAG accuracy (Assignment 4)

Research question: how does information recency (changed within 24 hours vs 2-7 days earlier)
affect the answer accuracy of RAG-based LLMs on time-sensitive questions?

Team 12, group SE-2414: Shakarim Ainatayev, Kuanysh Seitzhan, Mukhammedali Khassenov.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Builds the frozen test set `questions_final.csv` (own PyPI questions + FreshQA + SimpleQA) and `dataset_info.json` (counts, SHA-256) |
| `rag_pipeline.py` | Runs the experiment: one Tavily search per question, three conditions (no retrieval, standard RAG, FreshPrompt), 3 repeats, automatic scoring, logs |
| `analyze.py` | Builds the main results table, statistical tests, efficiency table and figures from the results file |
| `error_analysis.py` | Error types (Appendix B), retrieval hit rate and sensitivity checks from the results file and the manual error labels |
| `questions_final.csv`, `dataset_info.json` | Frozen test set of the final run (348 questions) and its counts and SHA-256 |
| `a3_question_ids.txt` | Questions used in Assignment 3 (development data), excluded from the test set |
| `questions_test.csv` | 4 development questions for a quick check |
| `results/` | Final run `20261108_133215` (8 November 2026, 18:32-20:05 Astana time): raw log, results, summary, manual error labels (`error_labels.csv`) and `analysis/` |

## Installation

Python 3.11, then:

```
pip install -r requirements.txt
```

Copy `keys.env.example` to `keys.env` and write your River and Tavily API keys into it.
`keys.env` is listed in `.gitignore` and is never uploaded.

## Reproduce the main results table from the logs (no API keys needed)

```
python analyze.py results/results_20261108_133215.csv
python error_analysis.py results/results_20261108_133215.csv results/error_labels.csv
```

Output goes to `results/analysis/`: main_table.csv, stats.csv, efficiency.csv, error_cases.csv,
error_summary.csv, error_by_subset.csv, retrieval.csv, sensitivity.csv and the figures.
`error_labels.csv` holds the manual category of every question standard RAG answered wrongly;
it was made from `error_cases.csv` by reading the raw log.

## Re-run the experiment (API keys needed)

```
python rag_pipeline.py questions_test.csv
python rag_pipeline.py questions_final.csv
python analyze.py results/results_<run_id>.csv
```

The own questions depend on the moment they are built, so a new `build_dataset.py` run produces
new questions. To repeat the exact study, use the committed `questions_final.csv`
(its SHA-256 is printed by the pipeline and must match `dataset_info.json`).
Model answers and web search results can change over time, so a re-run may differ slightly.
