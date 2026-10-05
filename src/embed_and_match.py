"""Recompute the deployment scores from scratch. Optional - costs API calls.

    python src/embed_and_match.py

You do not need this to reproduce any figure or table: the scores it produces
are already committed as data/agent_task_best_matches_openai_mkt.csv, and the
figure and table scripts read that file. Run this only to re-derive the scores,
for example against a different provider corpus or a newer embedding model.

What it does, following task_3_embed_dist_openai.ipynb:

  1. Take the task text as-is. The source notebook defines a spaCy cleaner
     (stopwords, punctuation, digits and very short tokens removed, lowercased)
     but disables it - "do not clean the text for now" - and embeds the raw
     text. The committed cleaned_task column is a verbatim copy of Task, which
     confirms it. Pass --clean to apply the cleaner instead; the scores will not
     match the committed ones if you do.
  2. Embed both sides with OpenAI text-embedding-3-large.
  3. For each O*NET task, take the cosine similarity against every provider
     description; keep the best one as `best_match` / `best_match_score` and the
     average as `mean_match_score`.

Embeddings are cached to .cache/*.pkl, so a re-run costs nothing after the first.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BATCH_SIZE, DATA, EMBEDDING_MODEL, ONET_TASKS, PROVIDER_TASKS, ROOT,
    SPACY_MODEL,
)

CACHE = ROOT / ".cache"
OUT = DATA / "agent_task_best_matches_openai_mkt.recomputed.csv"


def clean_text(nlp, text):
    doc = nlp(str(text))
    toks = [t.text for t in doc if not t.is_stop and not t.is_punct]
    toks = [t.lower() for t in toks if t.isalpha() and len(t) > 2]
    return " ".join(toks)


def get_embeddings(client, texts, cache_path):
    if cache_path.exists():
        print(f"  cache hit: {cache_path.name}")
        return np.array(pickle.load(cache_path.open("rb")))
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend(e.embedding for e in resp.data)
        print(f"  embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    cache_path.parent.mkdir(exist_ok=True)
    pickle.dump(out, cache_path.open("wb"))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description="Recompute the deployment scores.")
    ap.add_argument("--clean", action="store_true",
                    help="apply the spaCy cleaner (the original run did not)")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export the variable. See README."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        sys.exit(f"{e}. Install it with: pip install openai")

    nlp = None
    if args.clean:
        import spacy
        nlp = spacy.load(SPACY_MODEL)
    client = OpenAI()

    onet = pd.read_csv(ONET_TASKS)
    providers = pd.read_csv(PROVIDER_TASKS)
    print(f"O*NET tasks: {len(onet)} | provider descriptions: {len(providers)} "
          f"({providers['service_name'].nunique()} services)")

    if nlp is None:
        print("using raw task text (as the original run did)")
        onet["cleaned_task"] = onet["Task"]
        providers["cleaned_task"] = providers["task_involved"]
    else:
        print("applying the spaCy cleaner (scores will differ from the committed ones)")
        onet["cleaned_task"] = [clean_text(nlp, t) for t in onet["Task"]]
        providers["cleaned_task"] = [clean_text(nlp, t) for t in providers["task_involved"]]

    print("embedding provider descriptions...")
    emb_p = get_embeddings(client, providers["cleaned_task"].tolist(),
                           CACHE / "openai_embeddings_agent_task.pkl")
    print("embedding O*NET tasks...")
    emb_o = get_embeddings(client, onet["cleaned_task"].tolist(),
                           CACHE / "openai_embeddings_mkt_sales_task.pkl")

    # Cosine similarity: both sides L2-normalised, then a single matmul.
    emb_p = emb_p / np.linalg.norm(emb_p, axis=1, keepdims=True)
    emb_o = emb_o / np.linalg.norm(emb_o, axis=1, keepdims=True)
    sims = emb_o @ emb_p.T

    best = sims.argmax(axis=1)
    onet["best_match"] = providers["task_involved"].to_numpy()[best]
    onet["best_match_score"] = sims.max(axis=1)
    onet["mean_match_score"] = sims.mean(axis=1)

    onet.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("Compare against the committed scores before replacing them:")
    print(f"  python -c \"import pandas as pd; "
          f"a=pd.read_csv('data/agent_task_best_matches_openai_mkt.csv'); "
          f"b=pd.read_csv('{OUT.name}'); "
          f"print((a.best_match_score-b.best_match_score).abs().max())\"")


if __name__ == "__main__":
    main()
