# Delegating Marketing Tasks to AI Agents — replication package

Replication code and data for the task-based analysis of AI agent deployment
across marketing occupations: which O*NET marketing tasks have a close semantic
counterpart in the services AI agent providers advertise.

Everything reproduces from the committed scores with **no API key and no GPU**,
in under a minute.

---

## Reproduce everything

```bash
conda create -n taskdel python=3.12 -y && conda activate taskdel
pip install -r requirements.txt

python src/make_figures.py
python src/make_tables.py
```

| Output | Manuscript item |
|---|---|
| `figures/occupation_task_replacement_openai.png` | Fig. 2 — share of tasks above the cutoff, by occupation |
| `figures/subcluster_task_replacement_openai.png` | Fig. 3 — same, by sub-cluster |
| `figures/best_match_score_distribution_openai.png` | Appendix D — deployment-score distribution |
| `tables/thematic_summary.tex` | Table 1 — tasks by deployment-score tercile |
| `tables/onet_occupations.tex` | Appendix A — occupations by sub-cluster |
| `tables/providers.tex` | Appendix B — provider inventory |
| `tables/matched_providers.tex` | Providers selected as a best match, with task counts |

Figure filenames match the `\includegraphics` paths in the manuscript, so
`figures/` can be copied straight into the LaTeX project. The generated tables
reproduce the manuscript's own LaTeX layout — `tabularx` with `multirow` and
`makecell` for Table 1, `longtable` for the provider inventory — so they can be
pasted in directly. They need `booktabs`, `longtable`, `array`, `tabularx`,
`multirow` and `makecell` in the preamble.

## Method

Each of the 605 O*NET marketing tasks and each provider task description is
cleaned (stopwords, punctuation, digits and very short tokens removed, then
lowercased) and embedded with OpenAI `text-embedding-3-large`. For every O*NET
task, cosine similarity is taken against every provider description; the highest
is its **deployment score**. A task counts as deployed when that score is at
least **0.5**. Figures report the share of **Core** tasks above the cutoff.

All parameters live in `src/config.py`.

## Repository layout

```
data/
  Marketing_Sales_Tasks.csv                   605 O*NET tasks, 31 occupations, 4 sub-clusters
  ai_agent_task_solved_summary_combined.csv   provider task descriptions
  agent_task_best_matches_openai_mkt.csv      the 605 tasks with deployment scores (analysis input)
  thematic_categories.csv                     Table 1 category names
  provider_descriptions.csv                   Appendix B provider inventory
src/
  config.py            paths, the 0.5 cutoff, the Core filter
  make_figures.py      Fig. 2, Fig. 3, Appendix D
  make_tables.py       Table 1 and the appendix tables
  embed_and_match.py   optional: recompute scores from raw text
  refresh_onet.py      optional: apply task types from a newer O*NET release
figures/   tables/     generated output
```

### What is computed and what is an input

Computed from the data: the tercile assignment and cutpoints, every proportion
in Figs. 2 and 3, the occupation lists in Appendix A, and the provider counts in
`matched_providers.tex`.

An input: the five category names per tercile in Table 1, which were written
from the prompt documented in the manuscript appendix and ship as
`data/thematic_categories.csv`. The example task attached to each category is
checked against the data on every run.

## Optional: recompute the scores

Needs an OpenAI key and costs a few cents.

```bash
pip install -r requirements-match.txt
python -m spacy download en_core_web_sm
cp .env.example .env     # then add your key
python src/embed_and_match.py
```

Writes `data/agent_task_best_matches_openai_mkt.recomputed.csv` rather than
overwriting the committed scores, so the two can be compared. Embeddings are
cached under `.cache/`, so a second run is free.

## Optional: apply a newer O*NET release

O*NET updates task statements and their Core/Supplemental designations on a
rolling basis. To map the designations from a newer release onto the existing
scores:

```bash
# "Task Statements.txt" comes from the text bundle at onetcenter.org/database.html
python src/refresh_onet.py --task-statements "/path/to/Task Statements.txt"

TASKDEL_SCORES=data/agent_task_best_matches_openai_mkt.onet31.csv \
    python src/make_figures.py
```

Results for O*NET 31.0 are committed: the mapped scores as
`data/agent_task_best_matches_openai_mkt.onet31.csv` and the resulting figures
under `figures/onet31/`. The figures in `figures/` remain the default.

## Provenance

The analysis originates in two notebooks from the project's working directory:
`task_3_embed_dist_openai.ipynb` (cleaning, embedding, matching) and
`Visualization_tasks.ipynb` (figures). The scripts here are those notebooks made
runnable, with output paths inside the repository.

Figure styling follows `Visualization_tasks.ipynb`, with two adjustments: Fig. 3
uses a wider frame so the x-axis label fits, and Fig. 2 has an explicit
tie-break on occupation name so repeated runs are byte-identical. Several
occupations share a value, and tie order carries no meaning.
