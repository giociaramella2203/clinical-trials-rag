# Clinical Trials Eligibility Assistant

An agentic RAG system that answers questions about clinical trials using live
data from [ClinicalTrials.gov](https://clinicaltrials.gov), and can precisely
check age-based eligibility for a specific trial using a deterministic tool
call rather than relying on the LLM to reason about numbers from prose.
Includes a FastAPI service layer, a Streamlit chat interface, a test suite
with CI, and a Dockerfile for containerized deployment.

## What it does

- **General questions** ("What trials exist for rheumatoid arthritis?") are
  answered via retrieval: the system searches a local index of ~200 trials
  (pulled live from the ClinicalTrials.gov API) and generates an answer
  grounded in the most relevant matches.
- **Specific eligibility questions** ("Is a 70-year-old eligible for trial
  NCT07045896?") are routed to a dedicated tool call that looks up the exact
  trial and checks the patient's age against its stated min/max age
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
Retrieve (keyword search over local trial index)              ▼
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
specific trial ID that keyword search failed to retrieve (a near-inevitable
outcome, since IDs are poor keyword-search targets and common words like
"patient" or "eligible" dominate scoring across the whole corpus), the model
would see irrelevant context, conclude the question wasn't answerable, and
refuse — rather than falling back to the available tool.

**Fix:** detect an explicit NCT ID in the query *before* calling the model,
and skip retrieval entirely in that case. This removes the competing signal
(irrelevant context vs. tool-use instruction) rather than trying to make the
instruction "win" through stronger prompt wording — a more robust fix,
since it eliminates the ambiguity at the source instead of hoping the model
resolves it correctly every time.

## Tech stack

- **Data source:** [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) (free, no auth)
- **Retrieval:** [`minsearch`](https://github.com/alexeygrigorev/minsearch) — keyword search over trial titles, eligibility text, conditions, and NCT IDs
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

### Possible next steps

- Replace or supplement keyword search (`minsearch`) with embeddings-based
  semantic search, which would better handle queries that don't share exact
  wording with the underlying trial text (the current dataset size — ~200
  trials — doesn't strictly require this, but it's the natural next
  improvement toward a production-scale system).
- Broaden the eligibility tool beyond age (e.g. sex, condition-specific
  criteria) as additional structured fields become useful.

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