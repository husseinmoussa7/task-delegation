"""Shared paths and parameters for the task-delegation analysis.

Every number the manuscript reports comes from these three settings applied to
`data/agent_task_best_matches_openai_mkt.csv`. Changing any of them changes the
published figures, so they live here rather than being repeated in each script.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

# Inputs. TASKDEL_SCORES overrides the scores file, so an alternative version
# (for example the O*NET 31.0 refresh from src/refresh_onet.py) can be run
# through the same figure and table scripts without editing anything:
#   TASKDEL_SCORES=data/..._mkt.onet31.csv python src/make_figures.py
SCORED_TASKS = Path(os.getenv("TASKDEL_SCORES") or DATA / "agent_task_best_matches_openai_mkt.csv")
if not SCORED_TASKS.is_absolute():
    SCORED_TASKS = ROOT / SCORED_TASKS
ONET_TASKS = DATA / "Marketing_Sales_Tasks.csv"
PROVIDER_TASKS = DATA / "ai_agent_task_solved_summary_combined.csv"
THEMATIC_CATEGORIES = DATA / "thematic_categories.csv"

# A task counts as "deployed" when its best semantic match against the provider
# corpus scores at or above this cutoff. Stated in the manuscript as the 0.5
# semantic-alignment cutoff.
DEPLOYMENT_THRESHOLD = 0.5

# Only Core tasks are reported. O*NET also marks tasks Supplemental, and 40 of
# the 605 rows carry no TaskType at all; both are excluded from every figure.
TASK_TYPE = "Core"

# Re-running the match step (src/embed_and_match.py) needs these. They are not
# used by the figure or table scripts, which read the committed scores.
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 1000
SPACY_MODEL = "en_core_web_sm"


def load_scored_tasks():
    """The 605 scored tasks, with the deployment flag and tercile attached."""
    import pandas as pd

    df = pd.read_csv(SCORED_TASKS)
    df["AI_replaced"] = df["best_match_score"] >= DEPLOYMENT_THRESHOLD
    df["tercile"] = pd.qcut(
        df["best_match_score"], 3, labels=["Bottom", "Middle", "Top"]
    )
    return df


def core_tasks():
    """Core tasks only — the basis of every reported figure."""
    df = load_scored_tasks()
    return df[df["TaskType"] == TASK_TYPE]
