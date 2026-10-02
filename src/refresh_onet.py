"""Apply O*NET task types from a newer release to the existing scores.

    python src/refresh_onet.py --task-statements "/path/to/Task Statements.txt"

Get the file from https://www.onetcenter.org/database.html (the "text" bundle;
`Task Statements.txt` is inside it).

O*NET revises task statements and their Core/Supplemental designations on a
rolling basis. Deployment scores depend only on task text, so a release that
leaves the text unchanged does not require re-scoring - but the Core filter the
figures apply does depend on the designations.

This script maps task types from a newer release onto the existing scores by
Task ID and writes a parallel scores file, leaving the published one untouched.
Tasks whose ID is absent from the new release keep their original type; tasks
new to the release have no score and are not added, since scoring them needs the
embedding step.

Regenerate the figures and tables against the result with:

    TASKDEL_SCORES=data/agent_task_best_matches_openai_mkt.onet31.csv \\
        python src/make_figures.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DEPLOYMENT_THRESHOLD, ROOT, SCORED_TASKS

OUT = DATA / "agent_task_best_matches_openai_mkt.onet31.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-statements", required=True,
                    help='Path to "Task Statements.txt" from an O*NET text bundle')
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    new = pd.read_csv(args.task_statements, sep="\t", dtype=str)
    new.columns = [c.strip() for c in new.columns]
    old = pd.read_csv(SCORED_TASKS, dtype={"TaskID": str})

    types = new.set_index("Task ID")["Task Type"]
    old["TaskType_new"] = old["TaskID"].map(types)

    missing_id = int(old["TaskType_new"].isna().sum())
    was_blank = old["TaskType"].isna()
    filled = int((was_blank & old["TaskType_new"].notna()).sum())
    changed = int(
        (old["TaskType"].notna()
         & old["TaskType_new"].notna()
         & (old["TaskType"] != old["TaskType_new"])).sum()
    )

    codes = set(old["Code"])
    added = len(set(new[new["O*NET-SOC Code"].isin(codes)]["Task ID"]) - set(old["TaskID"]))

    print(f"O*NET release has {len(new)} task statements")
    print(f"repo tasks: {len(old)}   not present in the new release: {missing_id}")
    print(f"blank TaskType filled in: {filled}")
    print(f"TaskType reassigned (Core <-> Supplemental): {changed}")
    print(f"tasks new to the release, with no score, NOT added: {added}")

    # Keep the original type where the new release has nothing to say.
    old["TaskType"] = old["TaskType_new"].fillna(old["TaskType"])
    old = old.drop(columns=["TaskType_new"])
    old.to_csv(args.out, index=False)
    print(f"\nwrote {Path(args.out).relative_to(ROOT)}")

    # Impact on the reported proportions.
    cur = pd.read_csv(SCORED_TASKS)
    cur["AI_replaced"] = cur["best_match_score"] >= DEPLOYMENT_THRESHOLD
    upd = old.copy()
    upd["AI_replaced"] = upd["best_match_score"] >= DEPLOYMENT_THRESHOLD
    a = cur[cur.TaskType == "Core"].groupby("Occupation")["AI_replaced"].mean()
    b = upd[upd.TaskType == "Core"].groupby("Occupation")["AI_replaced"].mean()

    print(f"\nCore tasks: {int((cur.TaskType=='Core').sum())} -> "
          f"{int((upd.TaskType=='Core').sum())}")
    print(f"Fig. 2 bars: {a.notna().sum()} -> {b.notna().sum()}")
    moved = [(o, a.get(o), b[o]) for o in b.index
             if o not in a.index or abs(b[o] - a[o]) > 0.004]
    print(f"\noccupations whose proportion moves ({len(moved)}):")
    for o, x, y in sorted(moved, key=lambda t: -(t[2])):
        print(f"  {o[:54]:54s} {'--' if pd.isna(x) else f'{x:.2f}':>6s} -> {y:.2f}")

    s1 = cur[cur.TaskType == "Core"].groupby("Sub-Cluster")["AI_replaced"].mean()
    s2 = upd[upd.TaskType == "Core"].groupby("Sub-Cluster")["AI_replaced"].mean()
    print("\nFig. 3:")
    for k in s2.sort_values(ascending=False).index:
        print(f"  {k:40s} {s1[k]:.2f} -> {s2[k]:.2f}")


if __name__ == "__main__":
    main()
