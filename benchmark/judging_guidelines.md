# Judging guidelines (cs benchmark)

Grade each pooled passage for the query **as written**, using only the passage text.

| Grade | Meaning | Example (query: *"TF-IDF formula and inverse document frequency"*) |
|---|---|---|
| **2 — highly relevant** | The passage directly answers or explains the query; a student could learn the answer from it. | Gives the tf·idf weighting and defines idf = log N/df. |
| **1 — partly relevant** | On the right topic but incomplete, tangential, or only mentions the concept in passing. | Mentions tf-idf weighting as one feature of a ranking model. |
| **0 — not relevant** | Different topic, or only shares words with the query. | A networking passage that contains "frequency". |

Rules
1. Judge relevance to the *information need*, not word overlap. A passage that uses different words for the right idea is relevant.
2. Do not look at which system retrieved the passage or at its rank (the judging page hides both).
3. Code listings, figures or exercises count if they answer the need.
4. When unsure between two grades, pick the lower one and add a note in `docs/ASSUMPTIONS.md`.
5. Two people judge the same ~20% of queries (16 queries: 6 val + 10 test) independently, with their names entered on the judging page. Compute agreement with
   `python -c "from contextweave.evaluation.agreement import kappa_report; kappa_report()"`, then discuss and resolve disagreements before the final run.
6. Queries were written before looking at any system output (git commit `Benchmark queries written before any system output`). Do not edit them after seeing results.
