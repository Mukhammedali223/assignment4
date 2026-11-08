"""
RAG pipeline v4 (final experiment, Assignment 4):
"Evaluating the Effect of Information Recency on the Accuracy of RAG-Based LLMs
 for Time-Sensitive Questions"

For every question in the CSV file:
  1. RETRIEVE  - ONE web search with Tavily (top-k results); both RAG conditions get the same documents
  2. ANSWER in three conditions, each asked REPEATS times:
       baseline    - the same model, no search results (simple baseline)
       rag         - standard RAG: search results in the prompt, answer only from them (main condition)
       freshprompt - FreshPrompt-style RAG (Vu et al., 2024): the same results, each with its
                     publication date, sorted oldest -> newest, today's date given, and the
                     instruction to prefer the most recent evidence (literature baseline)
  3. SCORE     - 3 automatic methods: strict, normalized (PRIMARY), token F1
  4. ERRORS    - RAG/FreshPrompt wrong: true answer in the retrieved text?
                 yes -> generation_error, no -> retrieval_miss
  5. LOG       - raw jsonl + results csv (written after every question) + summary csv,
                 latency of every call in milliseconds

Run:
  pip install -r requirements.txt
  python rag_pipeline.py questions_final.csv
Then:
  python analyze.py results/results_<run_id>.csv
"""
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import openai
import requests
from openai import OpenAI

# ======================= CONFIG (all settings in one place) =======================
KEYS_FILE = "keys.env"   # RIVER_API_KEY=... and TAVILY_API_KEY=... (never uploaded to GitHub)

RIVER_BASE_URL = "https://api.river.ai/v1"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"   # pinned model version
TEMPERATURE = 0          # 0 = most deterministic answers
MAX_TOKENS = 300
REASONING_EFFORT = "none"  # DeepSeek on River: none / high / max
REPEATS = 3              # how many times each question is asked (stability check)

TOP_K = 5                # how many search results go into the prompt
SEARCH_DEPTH = "advanced"  # Tavily: basic = 1 credit, advanced = 2 credits
CHUNKS_PER_SOURCE = 3    # text snippets per search result
MAX_CONTEXT_CHARS = 12000  # safety limit for the prompt size

F1_THRESHOLD = 0.5       # token F1 needed to count an answer as correct
CONDITIONS = ["baseline", "rag", "freshprompt"]

PAUSE_SEC = 0.2          # small pause between API calls (rate limits)
OUT_DIR = "results"
ASTANA = timezone(timedelta(hours=5))
# ==================================================================================

SYSTEM_RAG = (
    "You answer factual questions using ONLY the search results given by the user. "
    "Do not use your own memory. If the search results do not contain the answer, "
    "set found=false and answer=\"NOT FOUND\". The answer field must contain ONLY the "
    "value (number, version, name or date), no explanations; put the page you took it "
    "from into source_url."
)
SYSTEM_BASELINE = (
    "You answer factual questions from your own knowledge. If you do not know the "
    "answer, set found=false and answer=\"NOT FOUND\". The answer field must contain "
    "ONLY the value (number, version, name or date), no explanations."
)
SYSTEM_FRESHPROMPT = (
    "Today is {today}. You answer factual questions using ONLY the search results given "
    "by the user. Each result has a publication date; the results are sorted from the "
    "oldest to the most recent. Information may have changed recently: when results "
    "disagree, trust the most recent one. Do not use your own memory. If the search "
    "results do not contain the answer, set found=false and answer=\"NOT FOUND\". The "
    "answer field must contain ONLY the value (number, version, name or date), no "
    "explanations; put the page you took it from into source_url."
)
ANSWER_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "found": {"type": "boolean"},
                "answer": {"type": "string"},
                "source_url": {"type": "string"},
            },
            "required": ["found", "answer", "source_url"],
            "additionalProperties": False,
        },
    },
}

RESULT_FIELDS = ["id", "dataset", "group", "condition", "repeat", "asked_at_utc",
                 "asked_at_astana", "hours_since_event", "in_window", "ground_truth", "answer", "found",
                 "source_url", "correct_strict", "correct_norm", "f1", "correct_f1",
                 "correct", "truth_in_context", "error_type", "prompt_tokens",
                 "completion_tokens", "latency_ms", "search_latency_ms"]


def get_key(name):
    """Read an API key from keys.env (or from an environment variable)."""
    path = next((p for p in [KEYS_FILE, KEYS_FILE + ".txt"] if os.path.exists(p)), None)
    if path:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                key, _, value = line.partition("=")
                if key.strip() == name and value.strip():
                    return value.strip().strip('"').strip("'")
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing key: write {name}=... into {KEYS_FILE}")
    return value


def now_utc():
    return datetime.now(timezone.utc)


def read_questions(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def aliases_of(ground_truth):
    return [a.strip() for a in ground_truth.split("|") if a.strip()]


# ------------------------------- 1. RETRIEVE ----------------------------------------
def search_web(query, tavily_key):
    """Web search with Tavily. Returns a list of {url, title, published, content}."""
    payload = {
        "query": query[:400],
        "max_results": TOP_K,
        "search_depth": SEARCH_DEPTH,
        "chunks_per_source": CHUNKS_PER_SOURCE,
        "include_published_date": True,
    }
    headers = {"Authorization": f"Bearer {tavily_key}"}
    for attempt in range(5):
        r = requests.post("https://api.tavily.com/search", json=payload,
                          headers=headers, timeout=60)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 10)))
            continue
        r.raise_for_status()
        results = r.json().get("results", [])
        return [{"url": x.get("url", ""), "title": x.get("title", ""),
                 "published": x.get("published_date"), "content": x.get("content", "")}
                for x in results]
    raise RuntimeError("Tavily: too many 429 errors")


# ------------------------------- 2. AUGMENT -----------------------------------------
def build_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] {d['title']}\nURL: {d['url']}\n{d['content']}")
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def doc_date(d):
    """Publication date as YYYY-MM-DD, or '' if the search engine did not give one."""
    try:
        return parsedate_to_datetime(d["published"]).strftime("%Y-%m-%d")
    except (TypeError, ValueError, KeyError):
        pass
    m = re.match(r"\d{4}-\d{2}-\d{2}", d.get("published") or "")
    return m.group(0) if m else ""


def build_fresh_context(docs):
    """FreshPrompt order: undated results first, then dated ones from oldest to newest,
    so the most recent evidence stands closest to the question."""
    ordered = sorted(docs, key=lambda d: doc_date(d) or "0000")
    parts = []
    for i, d in enumerate(ordered, 1):
        parts.append(f"[{i}] {d['title']}\nDate: {doc_date(d) or 'unknown'}\n"
                     f"URL: {d['url']}\n{d['content']}")
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


# ------------------------------- 3. GENERATE ----------------------------------------
def ask_model(client, system_prompt, user_prompt):
    """One call to DeepSeek. Returns the parsed JSON answer, raw text, token usage, latency."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format=ANSWER_FORMAT,
        extra_body={"reasoning_effort": REASONING_EFFORT},
    )
    latency_ms = round((time.perf_counter() - t0) * 1000)
    raw = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            parsed = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            parsed = {"found": False, "answer": raw[:200], "source_url": ""}
    usage = resp.usage.model_dump() if resp.usage else {}
    return parsed, raw, usage, latency_ms


# ------------------------------- 5. SCORE -------------------------------------------
def contains_value(text, truth, answer_type):
    """Strict method: the exact ground-truth value appears in the text."""
    text = (text or "").lower().replace("\u00a0", " ")
    truth = truth.strip().lower()
    variants = {truth}
    if answer_type == "number":
        variants.add(truth.replace(".", ","))
        text = re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", text)
    for v in variants:
        if re.search(r"(?<![0-9.,])" + re.escape(v) + r"(?![0-9])", text):
            return True
    return False


def normalize(text):
    """lower case, no accents, no articles, punctuation -> spaces (dots inside numbers stay)."""
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\bv(?=\d)", "", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"[^\w.]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def has_phrase(haystack, needle):
    return bool(needle) and re.search(r"(^| )" + re.escape(needle) + r"( |$)", haystack) is not None


def norm_match(answer, aliases):
    """Normalized method: an alias is inside the answer (a short unit after a number may be missing), or a non-numeric answer is inside an alias."""
    a = normalize(answer)
    if not a or a == "not found":
        return False
    for alias in aliases:
        n = normalize(alias)
        if has_phrase(a, n):
            return True
        unit = re.fullmatch(r"(\d[\d.]*) [a-z]{1,3}", n)
        if unit and has_phrase(a, unit.group(1)):
            return True
        if not re.fullmatch(r"[\d. ]+", a) and len(a) >= 3 and has_phrase(n, a):
            return True
    return False


def token_f1(answer, aliases):
    """F1 method: best word overlap between the answer and any alias (0..1)."""
    pred = normalize(answer).split()
    if not pred or pred == ["not", "found"]:
        return 0.0
    best = 0.0
    for alias in aliases:
        gold = normalize(alias).split()
        common = sum(min(pred.count(w), gold.count(w)) for w in set(pred))
        if common == 0:
            continue
        p, r = common / len(pred), common / len(gold)
        best = max(best, 2 * p * r / (p + r))
    return round(best, 3)


def score(answer, q):
    aliases = aliases_of(q["ground_truth"])
    strict = any(contains_value(answer, al, q["answer_type"]) for al in aliases)
    norm = norm_match(answer, aliases)
    f1 = token_f1(answer, aliases)
    return {"correct_strict": int(strict), "correct_norm": int(norm), "f1": f1,
            "correct_f1": int(f1 >= F1_THRESHOLD), "correct": int(norm)}


def hours_between(start_iso, end_dt):
    if not start_iso:
        return None
    start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    return round((end_dt - start).total_seconds() / 3600, 1)


def group_ok(group, hours):
    """Checks that the question really is in its recency window at query time."""
    if group == "recent_24h":
        return "unknown" if hours is None else ("yes" if 0 <= hours < 24 else "NO")
    if group == "older_2_7d":
        return "unknown" if hours is None else ("yes" if 48 <= hours <= 168 else "NO")
    return "n/a"


# ------------------------------- 7. SUMMARY -----------------------------------------
def summarize(rows):
    table = {}
    for r in rows:
        key = (r["dataset"], r["group"], r["condition"])
        t = table.setdefault(key, {"n": 0, "strict": 0, "norm": 0, "f1": 0})
        t["n"] += 1
        t["strict"] += r["correct_strict"]
        t["norm"] += r["correct_norm"]
        t["f1"] += r["correct_f1"]
    out = []
    for (ds, grp, cond), t in sorted(table.items()):
        out.append({"dataset": ds, "group": grp, "condition": cond, "n_answers": t["n"],
                    "acc_strict": round(t["strict"] / t["n"], 3),
                    "acc_norm": round(t["norm"] / t["n"], 3),
                    "acc_f1": round(t["f1"] / t["n"], 3)})
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_question(q, client, tavily_key, log):
    asked_at = now_utc()
    hours = hours_between(q.get("t_event_utc"), asked_at)
    window = group_ok(q["group"], hours)
    print(f"[{q['id']}] dataset={q['dataset']} group={q['group']} "
          f"hours_since_event={hours} in_window={window}")
    print("Q:", q["question"])
    print("Ground truth:", q["ground_truth"])

    query = (q.get("search_query") or "").strip() or q["question"]
    t0 = time.perf_counter()
    docs = search_web(query, tavily_key)
    search_ms = round((time.perf_counter() - t0) * 1000)
    context = build_context(docs)
    truth_in_context = norm_match(context, aliases_of(q["ground_truth"]))
    print(f"Retrieved {len(docs)} results, ground truth in retrieved text: {truth_in_context}")
    log.write(json.dumps({"type": "retrieval", "id": q["id"], "asked_at": asked_at.isoformat(),
                          "search_query": query, "docs": docs}, ensure_ascii=False) + "\n")

    rows = []
    today = asked_at.astimezone(ASTANA).strftime("%Y-%m-%d")
    prompts = {
        "baseline": (SYSTEM_BASELINE, q["question"]),
        "rag": (SYSTEM_RAG, f"Search results:\n\n{context}\n\nQuestion: {q['question']}"),
        "freshprompt": (SYSTEM_FRESHPROMPT.format(today=today),
                        f"Search results:\n\n{build_fresh_context(docs)}\n\n"
                        f"Question: {q['question']}"),
    }
    for condition in CONDITIONS:
        system, prompt = prompts[condition]
        for rep in range(1, REPEATS + 1):
            parsed, raw, usage, latency_ms = ask_model(client, system, prompt)
            answer = str(parsed.get("answer", ""))
            s = score(answer, q)
            if s["correct"]:
                error_type = "none"
            elif condition == "baseline":
                error_type = "no_context"
            else:
                error_type = "generation_error" if truth_in_context else "retrieval_miss"
            row = {"id": q["id"], "dataset": q["dataset"], "group": q["group"],
                   "condition": condition, "repeat": rep,
                   "asked_at_utc": asked_at.strftime("%Y-%m-%d %H:%M:%S"),
                   "asked_at_astana": asked_at.astimezone(ASTANA).strftime("%Y-%m-%d %H:%M:%S"),
                   "hours_since_event": hours, "in_window": window,
                   "ground_truth": q["ground_truth"], "answer": answer,
                   "found": parsed.get("found"), "source_url": parsed.get("source_url", ""),
                   **s,
                   "truth_in_context": "" if condition == "baseline" else int(truth_in_context),
                   "error_type": error_type,
                   "prompt_tokens": usage.get("prompt_tokens"),
                   "completion_tokens": usage.get("completion_tokens"),
                   "latency_ms": latency_ms, "search_latency_ms": search_ms}
            rows.append(row)
            log.write(json.dumps({"type": "answer", **row, "raw_output": raw},
                                 ensure_ascii=False) + "\n")
            print(f"   {condition:11s} run {rep}: answer={answer[:40]!r:44s} "
                  f"strict={s['correct_strict']} norm={s['correct_norm']} f1={s['f1']:.2f}  "
                  f"error={error_type}")
            time.sleep(PAUSE_SEC)
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    questions_file = sys.argv[1] if len(sys.argv) > 1 else "questions_test.csv"
    with open(questions_file, "rb") as f:
        questions_sha256 = hashlib.sha256(f.read()).hexdigest()
    river_key = get_key("RIVER_API_KEY")
    tavily_key = get_key("TAVILY_API_KEY")
    client = OpenAI(api_key=river_key, base_url=RIVER_BASE_URL, timeout=120, max_retries=5)

    os.makedirs(OUT_DIR, exist_ok=True)
    run_id = now_utc().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(OUT_DIR, f"raw_log_{run_id}.jsonl")
    results_path = os.path.join(OUT_DIR, f"results_{run_id}.csv")
    settings = {"run_id": run_id, "questions_file": questions_file,
                "questions_sha256": questions_sha256, "conditions": CONDITIONS, "model": MODEL,
                "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                "reasoning_effort": REASONING_EFFORT, "repeats": REPEATS, "top_k": TOP_K,
                "search_depth": SEARCH_DEPTH, "chunks_per_source": CHUNKS_PER_SOURCE,
                "f1_threshold": F1_THRESHOLD,
                "openai_sdk": openai.__version__, "python": sys.version.split()[0]}
    print("SETTINGS:", json.dumps(settings, ensure_ascii=False))

    questions = read_questions(questions_file)
    for q in questions:
        q.setdefault("dataset", "own")
    rows, failed = [], []
    started = time.time()
    with open(raw_path, "w", encoding="utf-8") as log, \
         open(results_path, "w", encoding="utf-8-sig", newline="") as res:
        writer = csv.DictWriter(res, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        log.write(json.dumps({"type": "settings", **settings}, ensure_ascii=False) + "\n")
        for n, q in enumerate(questions, 1):
            elapsed = (time.time() - started) / 60
            print("\n" + "=" * 78 + f"\nQuestion {n}/{len(questions)}   ({elapsed:.1f} min)")
            try:
                q_rows = run_question(q, client, tavily_key, log)
            except Exception as e:
                print(f"   ERROR, question skipped: {type(e).__name__}: {e}")
                log.write(json.dumps({"type": "error", "id": q["id"], "error": str(e)},
                                     ensure_ascii=False) + "\n")
                failed.append(q["id"])
                continue
            rows.extend(q_rows)
            writer.writerows(q_rows)
            res.flush()
            log.flush()

    summary = summarize(rows)
    write_csv(os.path.join(OUT_DIR, f"summary_{run_id}.csv"), summary)

    print("\n" + "=" * 78 + "\nSUMMARY (accuracy = correct answers / all answers)")
    print(f"{'dataset':9s} {'group':15s} {'condition':11s} {'answers':>7s} "
          f"{'strict':>7s} {'norm':>7s} {'f1':>7s}")
    for s in summary:
        print(f"{s['dataset']:9s} {s['group']:15s} {s['condition']:11s} {s['n_answers']:7d} "
              f"{s['acc_strict']:7.3f} {s['acc_norm']:7.3f} {s['acc_f1']:7.3f}")
    errors = {}
    for r in rows:
        if r["condition"] != "baseline" and not r["correct"]:
            key = f"{r['condition']}:{r['dataset']}:{r['error_type']}"
            errors[key] = errors.get(key, 0) + 1
    print("RAG / FreshPrompt errors by type:", errors or "none")
    out_window = sorted({r["id"] for r in rows if r["in_window"] == "NO"})
    print("Questions outside their time window:", out_window or "none")
    print("Skipped questions (errors):", failed or "none")
    print(f"Total time: {(time.time() - started) / 60:.1f} min")
    print(f"Questions file SHA-256: {questions_sha256}")
    print(f"\nFiles saved in '{OUT_DIR}' folder (run {run_id})")


if __name__ == "__main__":
    main()
