diff --git a/README.md b/README.md
index af924fe..95e1106 100644
--- a/README.md
+++ b/README.md
@@ -10,12 +10,14 @@ with CI, and a Dockerfile for containerized deployment.
 ## What it does
 
 - **General questions** ("What trials exist for rheumatoid arthritis?") are
-  answered via retrieval: the system searches a local index of ~200 trials
-  (pulled live from the ClinicalTrials.gov API) and generates an answer
-  grounded in the most relevant matches.
+  answered via retrieval: the system embeds a local index of ~800 trials
+  across 8 conditions (pulled live from the ClinicalTrials.gov API) with a
+  local sentence-transformers model and ranks them by cosine similarity to
+  the query, then generates an answer grounded in the most relevant matches.
 - **Specific eligibility questions** ("Is a 70-year-old eligible for trial
-  NCT07045896?") are routed to a dedicated tool call that looks up the exact
-  trial and checks the patient's age against its stated min/max age
+  NCT07045896?", or "Is a 40-year-old woman eligible for NCT07045896?") are
+  routed to a dedicated tool call that looks up the exact trial and checks
+  the patient's age — and sex, if mentioned — against its stated
   requirements — a deterministic calculation, not a language-model guess.
 - Both paths are available in the same conversation, through a chat interface
   that keeps history, so a general question and a specific follow-up
@@ -30,7 +32,7 @@ User query
 Does the query mention a specific NCT ID? ──── yes ──▶ Tool-enabled call
     │ no                                                (check_eligibility)
     ▼                                                        │
-Retrieve (keyword search over local trial index)              ▼
+Retrieve (embedding similarity over local trial index)         ▼
     │                                                   Deterministic result
     ▼                                                        │
 Generate (LLM answers from retrieved context)                 ▼
@@ -49,11 +51,13 @@ or a separate frontend — would.
 An earlier version always injected retrieved context into every prompt,
 regardless of query type, while also giving the model access to the
 eligibility-checking tool. This created a genuine bug: when a query named a
-specific trial ID that keyword search failed to retrieve (a near-inevitable
-outcome, since IDs are poor keyword-search targets and common words like
-"patient" or "eligible" dominate scoring across the whole corpus), the model
-would see irrelevant context, conclude the question wasn't answerable, and
-refuse — rather than falling back to the available tool.
+specific trial ID that retrieval failed to surface (a near-inevitable
+outcome under keyword search, since IDs are poor keyword-search targets and
+common words like "patient" or "eligible" dominate scoring across the whole
+corpus — and still an unreliable signal under semantic search, since an NCT
+ID is not semantically similar to trial content), the model would see
+irrelevant context, conclude the question wasn't answerable, and refuse —
+rather than falling back to the available tool.
 
 **Fix:** detect an explicit NCT ID in the query *before* calling the model,
 and skip retrieval entirely in that case. This removes the competing signal
@@ -65,7 +69,7 @@ resolves it correctly every time.
 ## Tech stack
 
 - **Data source:** [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) (free, no auth)
-- **Retrieval:** [`minsearch`](https://github.com/alexeygrigorev/minsearch) — keyword search over trial titles, eligibility text, conditions, and NCT IDs
+- **Retrieval:** local semantic search via [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`) — each trial's title, eligibility criteria, and conditions are embedded once and compared to the query embedding by cosine similarity; embeddings are cached to disk (`trial_embeddings_cache.npz`) and only recomputed when the underlying trial data changes, so `--reload` restarts don't re-embed from scratch
 - **LLM:** `openai/gpt-oss-120b` served via [Groq](https://groq.com) (OpenAI-compatible API)
 - **Tool-calling:** OpenAI-style function calling for deterministic eligibility checks
 - **API layer:** [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
@@ -115,6 +119,9 @@ This runs the FastAPI service in a container. The Streamlit UI still runs
 locally (via `uv run streamlit run ui.py`) and talks to the containerized API
 the same way it talks to the locally-run one.
 
+The image installs `libgomp1`, a system library required by `torch` (a
+`sentence-transformers` dependency) on Debian slim images.
+
 ### Running tests
 
 ```bash
@@ -134,16 +141,26 @@ would make tests slow, costly, and non-deterministic.
 - [x] Test suite + CI
 - [x] Docker containerization
 - [x] Streamlit chat UI
+- [x] Embeddings-based semantic retrieval with disk-cached embeddings
+- [x] Dataset expanded to 8 conditions across oncology, chronic disease,
+      mental health, and cardiovascular domains
+- [x] Sex-based eligibility checking
 
 ### Possible next steps
 
-- Replace or supplement keyword search (`minsearch`) with embeddings-based
-  semantic search, which would better handle queries that don't share exact
-  wording with the underlying trial text (the current dataset size — ~200
-  trials — doesn't strictly require this, but it's the natural next
-  improvement toward a production-scale system).
-- Broaden the eligibility tool beyond age (e.g. sex, condition-specific
-  criteria) as additional structured fields become useful.
+- Broaden the eligibility tool beyond age and sex (e.g. condition-specific
+  or biomarker-based criteria) as additional structured fields become
+  useful.
+- Cross-trial comparative queries ("compare eligibility across the breast
+  cancer trials") currently rely on passing more retrieved results (5
+  instead of 3) into a single prompt rather than a dedicated comparison
+  feature. This works reasonably for a handful of trials but hasn't been
+  stress-tested against larger result sets — worth watching for the model
+  conflating details between trials as the retrieved set grows.
+- Invalidate/refresh the embeddings cache automatically when trial data is
+  intentionally updated (currently it's driven purely by a content hash, so
+  it self-heals on any real data change, but there's no explicit
+  cache-busting command).
 
 ## Notes
 
