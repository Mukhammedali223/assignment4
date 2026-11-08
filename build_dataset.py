"""
Builds the frozen test set for Assignment 4.

  own       our recency questions, collected automatically from PyPI right now:
            recent_24h  - latest stable release uploaded less than 20 hours ago
            older_2_7d  - latest stable release uploaded 50-160 hours ago
            (margins keep every question inside its window during the run:
             <24 h and 48-168 h at the moment the question is asked)
            Ground truth = that version number, taken from the official PyPI JSON API.
            Every older package is matched to a recent package with the closest
            popularity rank, so both groups are equally well known.
  freshqa   140 FreshQA questions (70 never-changing + 70 slow-changing, TEST and DEV
            splits; after removing the A3 questions TEST alone has too few)
  simpleqa  140 SimpleQA questions
  Questions used in Assignment 3 (development data) are excluded.

Output: questions_final.csv + dataset_info.json (counts and SHA-256 hash of the csv).

Run:
  python build_dataset.py
"""
import concurrent.futures
import csv
import hashlib
import io
import json
import random
from datetime import datetime, timedelta, timezone

import requests
from packaging.version import InvalidVersion, Version

SEED = 2026
N_OWN_PER_GROUP = 40
CANDIDATE_POOL = 5000
RECENT_MAX_H = 20
OLDER_MIN_H, OLDER_MAX_H = 50, 160
N_FRESHQA_PER_TYPE = 70
N_SIMPLEQA = 140

TOP_PACKAGES_URL = "https://hugovk.dev/top-pypi-packages/top-pypi-packages-30-days.min.json"
FRESHQA_URL = ("https://docs.google.com/spreadsheets/d/"
               "1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM/export?format=csv")
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
ASTANA = timezone(timedelta(hours=5))

OUT_FILE = "questions_final.csv"
FIELDS = ["id", "dataset", "group", "question", "search_query", "ground_truth",
          "answer_type", "t_event_utc", "source_url", "note"]

session = requests.Session()


def latest_stable(package):
    """Latest stable (not pre-release, not yanked) version and its first upload time."""
    try:
        data = session.get(f"https://pypi.org/pypi/{package}/json", timeout=30).json()
        best = None
        for version, files in data["releases"].items():
            files = [f for f in files if not f.get("yanked")]
            if not files:
                continue
            try:
                parsed = Version(version)
            except InvalidVersion:
                continue
            if parsed.is_prerelease or parsed.is_devrelease:
                continue
            if best is None or parsed > best[0]:
                best = (parsed, version, min(f["upload_time_iso_8601"] for f in files))
        if best is None:
            return None
        uploaded = datetime.fromisoformat(best[2].replace("Z", "+00:00"))
        return best[1], uploaded
    except Exception:
        return None


def family(package):
    """Packages from one family (nab, nab-index, ...) count as one."""
    return package.lower().replace("_", "-").split("-")[0]


def load_own(now):
    top = session.get(TOP_PACKAGES_URL, timeout=60).json()["rows"][:CANDIDATE_POOL]
    names = [r["project"] for r in top if not r["project"].startswith("types-")]
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        info = list(pool.map(latest_stable, names))

    recent, older = [], []
    for rank, (name, item) in enumerate(zip(names, info)):
        if item is None:
            continue
        version, uploaded = item
        hours = (now - uploaded).total_seconds() / 3600
        if 0 <= hours < RECENT_MAX_H:
            recent.append((rank, name, version, uploaded))
        elif OLDER_MIN_H <= hours <= OLDER_MAX_H:
            older.append((rank, name, version, uploaded))

    used_families = set()
    chosen_recent = []
    for cand in recent:
        if family(cand[1]) not in used_families and len(chosen_recent) < N_OWN_PER_GROUP:
            chosen_recent.append(cand)
            used_families.add(family(cand[1]))

    chosen_older = []
    for rank, *_ in chosen_recent:
        free = [c for c in older if family(c[1]) not in used_families]
        if not free:
            break
        match = min(free, key=lambda c: abs(c[0] - rank))
        chosen_older.append(match)
        used_families.add(family(match[1]))

    date_text = now.astimezone(ASTANA).strftime("%d %B %Y").lstrip("0")
    rows = []
    for prefix, group, items in [("R", "recent_24h", chosen_recent), ("O", "older_2_7d", chosen_older)]:
        for i, (rank, name, version, uploaded) in enumerate(items, 1):
            rows.append({
                "id": f"{prefix}{i:02d}", "dataset": "own", "group": group,
                "question": f"As of {date_text}, what is the latest stable version of the "
                            f"{name} Python package published on PyPI?",
                "search_query": f"{name} python package latest version",
                "ground_truth": version, "answer_type": "text",
                "t_event_utc": uploaded.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_url": f"https://pypi.org/project/{name}/{version}/",
                "note": f"PyPI JSON API, popularity rank {rank + 1}"})
    return rows


def download_csv(url):
    r = session.get(url, timeout=120)
    r.raise_for_status()
    r.encoding = "utf-8"
    return list(csv.reader(io.StringIO(r.text)))


def load_freshqa(excluded):
    rows = download_csv(FRESHQA_URL)
    header_pos = next(i for i, r in enumerate(rows) if r and r[0] == "id")
    header = rows[header_pos]
    data = [dict(zip(header, r)) for r in rows[header_pos + 1:] if r and r[0]]
    data = [d for d in data if d["split"] in ("TEST", "DEV") and d["false_premise"] == "FALSE"
            and f"F{int(d['id']):03d}" not in excluded]
    rng = random.Random(SEED)
    out = []
    for fact_type in ["never-changing", "slow-changing"]:
        pool = [d for d in data if d["fact_type"] == fact_type]
        for d in rng.sample(pool, N_FRESHQA_PER_TYPE):
            aliases = [d[f"answer_{i}"].strip() for i in range(10) if d.get(f"answer_{i}", "").strip()]
            out.append({"id": f"F{int(d['id']):03d}", "dataset": "freshqa", "group": fact_type,
                        "question": d["question"].strip(), "search_query": "",
                        "ground_truth": " | ".join(aliases), "answer_type": "text",
                        "t_event_utc": "", "source_url": d["source"].split("\n")[0].strip(),
                        "note": f"FreshQA {d['split']} split"})
    return out


def load_simpleqa(excluded):
    rows = download_csv(SIMPLEQA_URL)
    data = [dict(zip(rows[0], r)) for r in rows[1:] if r]
    candidates = [i for i in range(len(data)) if f"S{i:04d}" not in excluded]
    rng = random.Random(SEED)
    out = []
    for i in sorted(rng.sample(candidates, N_SIMPLEQA)):
        out.append({"id": f"S{i:04d}", "dataset": "simpleqa", "group": "static",
                    "question": data[i]["problem"].strip(), "search_query": "",
                    "ground_truth": data[i]["answer"].strip(), "answer_type": "text",
                    "t_event_utc": "", "source_url": "", "note": "SimpleQA test set"})
    return out


def main():
    now = datetime.now(timezone.utc)
    with open("a3_question_ids.txt", encoding="utf-8") as f:
        excluded = {line.strip() for line in f if line.strip()}

    own = load_own(now)
    rows = own + load_freshqa(excluded) + load_simpleqa(excluded)
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    with open(OUT_FILE, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    counts = {}
    for r in rows:
        key = f"{r['dataset']}/{r['group']}"
        counts[key] = counts.get(key, 0) + 1
    info = {"file": OUT_FILE, "built_at_astana": now.astimezone(ASTANA).strftime("%Y-%m-%d %H:%M"),
            "total": len(rows), "counts": counts, "sha256": sha256, "seed": SEED}
    with open("dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))
    if len(own) < 2 * N_OWN_PER_GROUP:
        print(f"Note: only {len(own) // 2} own questions per group were available now.")


if __name__ == "__main__":
    main()
