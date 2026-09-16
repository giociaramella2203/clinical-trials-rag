# Clinical Trials Eligibility Assistant

An agentic RAG system that answers questions about clinical trials using live
data from [ClinicalTrials.gov](https://clinicaltrials.gov), and can precisely
check age-based eligibility for a specific trial using a deterministic tool
call rather than relying on the LLM to reason about numbers from prose.

## What it does

- **General questions** ("What trials exist for rheumatoid arthritis?") are
  answered via retrieval: the system searches a local index of ~200 trials
  (pulled live from the ClinicalTrials.gov API) and generates an answer
  grounded in the most relevant matches.
- **Specific eligibility questions** ("Is a 70-year-old eligible for trial
  NCT07045896?") are routed to a dedicated tool call that looks up the exact
  trial and checks the patient's age against its stated min/max age
  requirements — a deterministic calculation, not a language-model guess.

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
- **Retrieval:** [`minsearch`](https://github.com/alexeygrigorev/minsearch) — keyword search over trial titles, eligibility text, and conditions
- **LLM:** `openai/gpt-oss-120b` served via [Groq](https://groq.com) (OpenAI-compatible API)
- **Tool-calling:** OpenAI-style function calling for deterministic eligibility checks
- **Environment:** [`uv`](https://github.com/astral-sh/uv) for dependency management

## Setup

```bash
uv sync
```

Create a `.env` file with:
```
GROQ_API_KEY=your_key_here
```

Run the notebook: `01_first_trial.ipynb`

## Status

- [x] Core retrieval + generation (RAG)
- [x] Agentic tool-use layer for deterministic eligibility checks
- [x] Routing fix for context/tool-use conflict
- [ ] FastAPI service wrapper
- [ ] Test suite + CI
- [ ] Docker containerization

## Notes

This project intentionally works with public, aggregate trial metadata only
— no patient records or personal health data are involved anywhere in the
system. "Patient age" is a hypothetical value supplied at query time, used
only to check against a trial's published eligibility criteria.
