# Assumptions and decisions

Written during the build (checkpoints C0–C6). Anything here can be changed by the team; each item says where.

## Build process
- **All checkpoints C0–C6 were built in one go** at the requester's instruction ("build this in detail"), instead of
  stopping after each checkpoint. Tests were run after every checkpoint; the per-checkpoint report is in the README.
- **Optional extensions of spec §13 were not built** (positional index, compression, spelling, tiered index, BM25F,
  Rocchio, HNSW, LSI, re-ranking, learned router, calibrated confidence, RRF). They are left for M1–M4.
- **Novel extensions were added behind off-by-default flags** (`features:` in `config/settings.yaml`): QPP confidence
  signals, value-of-computation escalation, Learn-then-Test risk-controlled τ, QPP-percentile fusion. With all flags off
  the system is exactly the specification. They were chosen to be distinct from the §13 list so each member still has
  extensions to build.

## Corpus (C1)
- Sources (all openly available, listed in `data/sources.csv`, per-document list in `data/manifest.csv`):
  OSTEP (OS), Erickson *Algorithms* + notes (Algorithms), CMU 15-445 Fall 2022 notes + *Readings in Database Systems*
  (DBMS), Dordal *An Introduction to Computer Networks* (Networks), Croft/Metzler/Strohman *Search Engines: IR in
  Practice* + 22 arXiv IR papers (IR). Wikipedia was not used (rate-limited from the build machine).
- 163 logical documents / 7,661 passages. Big textbooks are split into one document per chapter (PDF outline, or
  detected chapter heading pages for the Croft book, whose chapter titles are written in `sources.csv`).
  The spec targets 300–500 documents; fewer, longer documents were accepted ("fewer is fine").
- DBMS is the smallest domain (418 passages). Adding more DBMS sources is an easy M1 task.
- Cleaning: repeated header/footer lines (≥ 40 % of pages, among first/last 3 lines, digits masked), hyphenated line
  breaks joined, pages < 300 chars skipped, trailing "References/Bibliography" sections dropped, and — because some
  PDFs (notably the Croft book) lose inter-word spaces — alphabetic tokens ≥ 15 letters that split into ≥ 3 words are
  re-segmented with `wordninja` (`ingestion.split_glued_words`).
- Passages are 200 **whitespace words** with 40 words overlap; a passage's `page` is the page of its first word.
- Arxiv paper titles come from the arXiv API; other titles are written in `sources.csv`.

## Text processing
- NLTK's English stop-word list is **bundled** (`textproc/stopwords_en.txt`) instead of downloaded, so results are
  identical on every machine and no NLTK data download is needed. Porter stemmer from NLTK.
- Technical tokens (`tf-idf`, `802.11`, `c++`, `b+tree`) are kept intact and are not stemmed; hyphenated compounds
  additionally emit their stemmed parts so "tf idf" matches "tf-idf".

## Lexical (C2)
- BM25 default IDF is the Lucene variant `ln(1 + (N-df+0.5)/(df+0.5))` (never negative). The `okapi` variant
  (rank-bm25's BM25Okapi, negative IDF floored at 0.25·mean IDF) is implemented too and is the one the test compares
  with rank-bm25 score-for-score (`bm25.idf_variant`).
- Query term weight = term count in the query (as rank-bm25) plus expansion weights.
- TF-IDF uses log-tf × idf on both sides with cosine normalisation (ltc.ltc).
- `bm25_champion` = Champion Lists **and** Index Elimination (system S3). Champions are chosen by BM25 term
  contribution, r = 50. `bm25_ie` (Index Elimination only) is also registered for ablations.
  If filtering leaves fewer than k candidates, the min-match requirement is relaxed to 1 (logged in `last_stats`).
- Boolean retrieval is pure Boolean: matches are ordered by coordination level, then corpus order. Plain queries are
  treated as the AND of their terms (S0). Quoted phrases are the AND of their terms (no positional index yet — M1).

## Semantic (C3)
- Passages are embedded as `title. text` for the cs corpus (BEIR passages already contain their title).
- Query vectors are computed once per query and reused by dense search and expansion.
- Expansion candidates are terms with the highest summed log-tf·idf over the top-5 feedback passages; each candidate's
  **most frequent surface word** is embedded for the drift filter (stems like "retriev" embed badly). Threshold
  cosine ≥ 0.25, up to 5 terms, weight 0.3. Inside the controller the feedback is the current fused list.

## Controller (C4)
- Coverage = mean over the top-5 passages of the **IDF-weighted** share of query terms they contain (plain share is
  dominated by generic words). Margin = 1 − (10th normalised score) (the top normalised score is 1 after min-max).
- The loop follows §3 step 5 ("take the next affordable step"): an unaffordable step is **skipped** and logged, and
  the loop continues with later, cheaper steps. (The §9 pseudo-code breaks instead; switching is a one-line change.)
- The first method always runs, even if its cost exceeds the budget.
- Fusion uses the most complete lexical list available (expansion > bm25 > bm25_ie > bm25_champion > tfidf) as L.
- α(q) rules: exact (quotes/Boolean/formula or short high-IDF) / semantic (question, ≥ 8 terms, many unknown) /
  default. **The three α values are tuned on validation queries** together with τ and the budget, because the
  hand-set 0.7/0.5/0.3 underweighted BM25 on SciFact (full-escalation val NDCG@10 0.726 vs 0.747 tuned). The fixed
  hybrid S5 always uses α = 0.5.
- Router class "exact_short" (≤ 4 terms, mean IDF ≥ 3.0, no question word) uses the exact α.

## Costs and budget
- Costs are measured (`make calibrate-costs`): 1 unit = median warm full-BM25 time on validation queries; recorded
  per collection in `costs_by_collection`. Expansion is timed as it runs inside the ladder (query vector already
  computed, feedback given).
- Finding: with vectorised numpy accumulation over precomputed impacts, full BM25 takes ~0.1–0.4 ms, so Champion Lists
  (≈1.2 units) are **not cheaper in wall-clock** even though they touch far fewer postings; dense retrieval is
  23–59 units (query encoding dominates). Reported, not hidden.
- Budget grid for tuning = every total cost reachable by a subset of the ladder methods.

## Evaluation (C5)
- **Tuning objective** (validation only): for each adaptive system choose the cheapest (τ, budget) whose mean NDCG@10
  is within 0.01 of the fixed hybrid on validation; if none, the best one.
- Public test collections with human judgements are used in addition to the cs corpus: BEIR SciFact (validation =
  300 sampled train queries, test = 300) and NFCorpus (validation = 300 dev queries, test = 323).
  For BEIR, "query type" in by-type results is the router's query class (BEIR has no human query types).
- The cs benchmark has 80 queries (27 exact, 27 semantic, 26 mixed; 30 val / 50 test), committed before any system
  output existed. **Its relevance labels must be made by the team** with the judging page (pool already built:
  1,542 passages). Until then, cs results cannot be computed and no cs numbers are reported anywhere.
- Latency: median and 95th percentile over up to 100 queries × 3 warm repetitions (median per query), on the machine
  recorded in every result file. Quality runs use memoised retrievers (exact, since retrievers are deterministic).
- Significance: paired t-test, Wilcoxon signed-rank and bootstrap 95 % CI on per-query NDCG@10 versus S5 and S2,
  Holm-adjusted across systems.
- F1@10 is reported as the harmonic mean of mean P@10 and mean Recall@10 (accuracy is not meaningful for ranking).

## Novel extensions
- Learn-then-Test risk control: loss = 1[NDCG@10_controller < NDCG@10_S5 − ε], ε = 0.05, δ = 0.15, η = 0.10,
  fixed-sequence testing over τ from 0.95 down; exact binomial p-values. The budget is set non-binding so τ alone
  controls escalation. A δ-curve (0.05–0.30) is also stored and validated on the test split.
- Value-of-computation gains are fitted from an exhaustive simulation of every escalation order on the validation
  queries; λ is tuned with the same objective as τ.

## Web UI (replaces the Streamlit app)
- **Stack:** FastAPI (`api/`) + React/TypeScript/Tailwind built with Vite (`web/`), the same front-end stack as the
  team's other projects. `python main.py serve` serves both from one port; the UI is built once with `npm run build`.
- **Look:** follows the team's Clothing Search Engine UI (`ui_theme.py`): paper background, sage accent, Fraunces
  headings, Plus Jakarta Sans text. Light theme only, so projector demos and screenshots never come out inverted.
  The chart accent is a slightly more saturated sage (`#3f7a34`) that passes the colour-blind / contrast checks.
- **Six Streamlit tabs became three pages.** Lexical, Semantic and Controller-trace detail now sit under each search
  result ("decision path", "what each method retrieved", "confidence signals"); Analytics became Evaluation.
- **Interactive searches use the collection's tuned S8 τ and budget** (`experiments/results/models/<c>/tuned.json`),
  in the UI and in `main.py search`, so what you see matches the evaluated system. The cs corpus falls back to the
  config values until it is tuned.
- **The Evaluation page only reads `experiments/results/*.json`;** it never recomputes a number. The Boolean baseline
  (NDCG@10 ≈ 0.04) is listed under the chart rather than plotted, so it does not flatten the scale.
- **Each `System` owns its retrievers** (`System.retrievers`); the global registry is still filled for older code.
  The server keeps one System per collection and loads them on demand, sometimes concurrently.
- **Transformers ≥ 5 load report:** all-MiniLM-L6-v2 was saved with a `position_ids` buffer that newer transformers
  rebuild instead of loading, so it was reported as UNEXPECTED. Verified identical (0..511) and the embeddings
  unchanged (cosine 1.0 against the stored index); only that exact finding is silenced (`semantic/embedder.py`).
