# Clinical Trials Eligibility Assistant

An agentic RAG system that answers questions across ~800 clinical trials
spanning 8 conditions, using local semantic search (sentence-transformers
embeddings) over live data from [ClinicalTrials.gov](https://clinicaltrials.gov).
Precisely checks age- and sex-based eligibility for a specific trial using a
deterministic tool call rather than relying on the LLM to reason from prose.
Includes a FastAPI service layer, a Streamlit chat interface, an evaluation
harness, query logging, a test suite with CI, and Docker/cloud deployment.

## 🔗 Live demo

Try it here — no setup required: **[clinical-trials-rag on Streamlit](https://giociaramella2203-clinical-trials-rag-streamlit-app-3lb3dz.streamlit.app/)**

(Note: the first question after a period of inactivity may take a little longer, as the app spins back up and loads trial data.)

## What it does

- **General questions** ("What trials exist for rheumatoid arthritis?") are
  answered via retrieval: the system embeds a local index of ~800 trials
  across 8 conditions (pulled live from the ClinicalTrials.gov API) with a
  local sentence-transformers model and ranks them by cosine similarity to
  the query, then generates an answer grounded in the most relevant matches.
- **Specific eligibility questions** ("Is a 70-year-old eligible for trial
  NCT07045896?", or "Is a 40-year-old woman eligible for NCT07045896?") are
  routed to a dedicated tool call that looks up the exact trial and checks
  the patient's age — and sex, if mentioned — against its stated
  requirements — a deterministic calculation, not a language-model guess.
- Both paths are available in the same conversation, through a chat interface
  that keeps history, so a general question and a specific follow-up
  eligibility check can happen back to back.

## Architecture

```
User query
    │
    ▼
Does the query mention a specific NCT ID? ──── yes ──▶ Tool-enabled call
    │ no                                                (check_eligibility)
    ▼                                                        │
Retrieve (embedding similarity over local trial index)         ▼
    │                                                   Deterministic result
    ▼                                                        │
Generate (LLM answers from retrieved context)                 ▼
    │                                                   Final LLM call,
    ▼                                                   grounded in result
   Answer ◀─────────────────────────────────────────────────┘
```

This core logic (`rag_agentic` in `app.py`) is wrapped in a FastAPI service
exposing a single `POST /ask` endpoint. A Streamlit chat app (`ui.py`) talks
to that endpoint over HTTP, the same way any other client — a script, `curl`,
or a separate frontend — would.

## Why the routing step exists

An earlier version always injected retrieved context into every prompt,
regardless of query type, while also giving the model access to the
eligibility-checking tool. This created a genuine bug: when a query named a
specific trial ID that retrieval failed to surface (a near-inevitable
outcome under keyword search, since IDs are poor keyword-search targets and
common words like "patient" or "eligible" dominate scoring across the whole
corpus — and still an unreliable signal under semantic search, since an NCT
ID is not semantically similar to trial content), the model would see
irrelevant context, conclude the question wasn't answerable, and refuse —
rather than falling back to the available tool.

**Fix:** detect an explicit NCT ID in the query *before* calling the model,
and skip retrieval entirely in that case. This removes the competing signal
(irrelevant context vs. tool-use instruction) rather than trying to make the
instruction "win" through stronger prompt wording — a more robust fix,
since it eliminates the ambiguity at the source instead of hoping the model
resolves it correctly every time.

## A hallucination bug found via manual testing

The NCT-ID routing branch originally sent the model only the numeric
eligibility data returned by `check_eligibility` (age range, sex
requirement) — not the trial's actual title or condition. When asked to
describe a trial by ID, the model had no real information to draw on for
*what the trial studies*, and in one observed case fabricated a
plausible-sounding but entirely wrong description (inventing a multiple
myeloma antibody trial for what was actually a lung cancer pulmonary
rehabilitation study) — while still computing the age-eligibility answer
correctly, since that part *was* grounded in real data.

**Fix:** `check_eligibility` now also returns the trial's real title and
conditions, and the NCT-ID branch includes an explicit instruction not to
describe the trial beyond what the tool provides. This closes the gap
between "the eligibility math is correct" and "the surrounding description
is trustworthy" — both now draw on the same real data.

This was caught through manual testing, not the automated eval suite —
worth noting as a real limitation: `eval.py`'s eligibility checks verify
the *eligible/not-eligible* judgment but don't check whether the trial
description in the answer is factually accurate. A stronger eval would
also assert on the returned trial name/condition, not just the boolean
outcome.

## Tech stack

- **Data source:** [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) (free, no auth)
- **Retrieval:** local semantic search via [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`) — each trial's title, eligibility criteria, and conditions are embedded once and compared to the query embedding by cosine similarity; embeddings are cached to disk (`trial_embeddings_cache.npz`) and only recomputed when the underlying trial data changes, so `--reload` restarts don't re-embed from scratch
- **LLM:** `openai/gpt-oss-120b` served via [Groq](https://groq.com) (OpenAI-compatible API)
- **Tool-calling:** OpenAI-style function calling for deterministic eligibility checks
- **API layer:** [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **UI:** [Streamlit](https://streamlit.io/) chat interface
- **Testing:** [pytest](https://docs.pytest.org/), covering the deterministic logic (age parsing, eligibility checks) — not the live API/LLM calls
- **CI:** GitHub Actions, running the test suite on every push
- **Deployment:** Docker
- **Environment:** [`uv`](https://github.com/astral-sh/uv) for dependency management

## Setup

```bash
uv sync
```

Create a `.env` file with:
```
GROQ_API_KEY=your_key_here
```

## Running the app

1. Start the API:
```bash
   uv run uvicorn app:app --reload
```
2. In a separate terminal, start the UI:
```bash
   uv run streamlit run ui.py
```
3. Open the Streamlit URL shown in the terminal (usually `http://localhost:8501`)

The UI is a chat interface — ask general questions about trials, or ask about
a specific trial's eligibility by mentioning its NCT ID and a patient age.
Conversation history is kept within a session, so follow-up questions work
naturally (e.g. ask about trials for a condition, then ask whether a given
age is eligible for one of the trials mentioned).

### Running with Docker

```bash
docker build -t clinical-trials-rag .
docker run -p 8000:8000 --env-file .env clinical-trials-rag
```

This runs the FastAPI service in a container. The Streamlit UI still runs
locally (via `uv run streamlit run ui.py`) and talks to the containerized API
the same way it talks to the locally-run one.

The image installs `libgomp1`, a system library required by `torch` (a
`sentence-transformers` dependency) on Debian slim images.

### Running tests

```bash
uv run pytest tests/
```

Tests cover pure, deterministic logic only (age parsing, eligibility
checks) — not the live ClinicalTrials.gov API calls or LLM calls, which
would make tests slow, costly, and non-deterministic.

## Status

- [x] Core retrieval + generation (RAG)
- [x] Agentic tool-use layer for deterministic eligibility checks
- [x] Routing fix for context/tool-use conflict
- [x] FastAPI service wrapper
- [x] Test suite + CI
- [x] Docker containerization
- [x] Streamlit chat UI
- [x] Embeddings-based semantic retrieval with disk-cached embeddings
- [x] Dataset expanded to 8 conditions across oncology, chronic disease,
      mental health, and cardiovascular domains
- [x] Sex-based eligibility checking

### Possible next steps

- Broaden the eligibility tool beyond age and sex (e.g. condition-specific
  or biomarker-based criteria) as additional structured fields become
  useful.
- Cross-trial comparative queries ("compare eligibility across the breast
  cancer trials") rely on passing more retrieved results (5 instead of 3)
  into a single prompt rather than a dedicated comparison feature. Manual
  testing with 5 retrieved trials produced accurate, well-organized
  comparisons with no observed detail conflation — but this hasn't been
  evaluated systematically, so larger result sets or more trials sharing
  overlapping criteria could still be worth watching.
- Invalidate/refresh the embeddings cache automatically when trial data is
  intentionally updated (currently it's driven purely by a content hash, so
  it self-heals on any real data change, but there's no explicit
  cache-busting command).

## Notes

This project intentionally works with public, aggregate trial metadata only
— no patient records or personal health data are involved anywhere in the
system. "Patient age" is a hypothetical value supplied at query time, used
only to check against a trial's published eligibility criteria.

Since trial data is pulled live from the ClinicalTrials.gov API rather than
cached, retrieval results can differ across runs — a trial recruiting today
may no longer match the `RECRUITING` status filter days later as its status
changes upstream. This is expected behavior for a system built on a live
data source, not a bug, but worth being aware of when comparing outputs
across sessions.