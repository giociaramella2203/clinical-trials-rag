from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os
import re
import json
import hashlib
import requests
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trial_embeddings_cache.npz"
)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

def trial_text(trial):
    return f"{trial['title']} {trial['eligibility_criteria']} {trial['conditions']}"

def trials_hash(trials):
    joined = "\n".join(t["nct_id"] + "|" + trial_text(t) for t in trials)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

def load_or_compute_embeddings(trials):
    current_hash = trials_hash(trials)

    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        cached = np.load(EMBEDDINGS_CACHE_PATH, allow_pickle=True)
        if str(cached["hash"]) == current_hash:
            print("Loaded trial embeddings from cache.")
            return cached["embeddings"]

    print("Computing trial embeddings (cache missing or stale)...")
    model = get_embedding_model()
    texts = [trial_text(t) for t in trials]
    embeddings = model.encode(texts, normalize_embeddings=True)
    np.savez(EMBEDDINGS_CACHE_PATH, embeddings=embeddings, hash=current_hash)
    return embeddings

def search_trials(query, num_results=5):
    model = get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = trial_embeddings @ query_embedding
    top_indices = np.argsort(similarities)[::-1][:num_results]
    return [all_trials[i] for i in top_indices]

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
    return _client

def parse_study(study):
    protocol = study["protocolSection"]
    return {
        "nct_id": protocol["identificationModule"].get("nctId"),
        "title": protocol["identificationModule"].get("briefTitle"),
        "status": protocol["statusModule"].get("overallStatus"),
        "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
        "eligibility_criteria": protocol.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
        "sex": protocol.get("eligibilityModule", {}).get("sex"),
        "min_age": protocol.get("eligibilityModule", {}).get("minimumAge"),
        "max_age": protocol.get("eligibilityModule", {}).get("maximumAge"),
    }

def fetch_trials(condition, page_size=100):
    resp = requests.get(
        "https://clinicaltrials.gov/api/v2/studies",
        params={
            "query.cond": condition,
            "filter.overallStatus": "RECRUITING",
            "fields": "NCTId,BriefTitle,EligibilityCriteria,MinimumAge,MaximumAge,Sex,Condition,OverallStatus",
            "pageSize": page_size,
        },
    )
    return resp.json()["studies"]

all_trials = []
trial_embeddings = None

def load_data():
    global all_trials, trial_embeddings

    if all_trials:
        return

    conditions = [
        "rheumatoid arthritis",
        "breast cancer",
        "type 2 diabetes",
        "asthma",
        "depression",
        "hypertension",
        "Alzheimer's disease",
        "lung cancer",
    ]
    all_trials = []
    for cond in conditions:
        studies = fetch_trials(cond)
        all_trials.extend([parse_study(s) for s in studies])

    for t in all_trials:
        t["conditions"] = " ".join(t["conditions"]) if t["conditions"] else ""

    trial_embeddings = load_or_compute_embeddings(all_trials)

    print(f"Loaded {len(all_trials)} trials.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_data()
    yield
    print("Shutting down.")

app = FastAPI(title="Clinical Trials Eligibility Assistant", lifespan=lifespan)

def parse_age(age_str):
    if not age_str:
        return None
    match = re.search(r"\d+", age_str)
    return int(match.group()) if match else None

def check_eligibility(nct_id, patient_age, patient_sex=None):
    trial = next((t for t in all_trials if t["nct_id"] == nct_id), None)
    if not trial:
        return {"error": "trial not found"}

    min_age = parse_age(trial["min_age"])
    max_age = parse_age(trial["max_age"])
    trial_sex = trial.get("sex")

    eligible_by_age = True
    reasons = []
    if min_age is not None and patient_age < min_age:
        eligible_by_age = False
        reasons.append(f"patient age {patient_age} is below minimum age {min_age}")
    if max_age is not None and patient_age > max_age:
        eligible_by_age = False
        reasons.append(f"patient age {patient_age} is above maximum age {max_age}")

    eligible_by_sex = True
    if patient_sex is not None and trial_sex and trial_sex != "ALL":
        eligible_by_sex = patient_sex.upper() == trial_sex.upper()
        if not eligible_by_sex:
            reasons.append(
                f"patient sex {patient_sex.upper()} does not match trial requirement {trial_sex}"
            )

    return {
        "nct_id": nct_id,
        "eligible_by_age": eligible_by_age,
        "eligible_by_sex": eligible_by_sex,
        "min_age": min_age,
        "max_age": max_age,
        "required_sex": trial_sex,
        "reasons": reasons,
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": "Check if a patient is eligible for a specific clinical trial based on its age requirements, and optionally its sex requirement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string", "description": "The NCT ID of the trial"},
                    "patient_age": {"type": "integer", "description": "The patient's age in years"},
                    "patient_sex": {
                        "type": "string",
                        "enum": ["MALE", "FEMALE"],
                        "description": "The patient's sex, if known. Only needed if the trial has a sex-specific requirement; omit if not mentioned in the question.",
                    },
                },
                "required": ["nct_id", "patient_age"],
            },
        },
    }
]

def build_agentic_prompt(query, results):
    context = "\n\n".join(
        f"NCT ID: {r['nct_id']}\nTitle: {r['title']}\nEligibility: {r['eligibility_criteria']}"
        for r in results
    )
    return f"""You are a clinical trials assistant. You have access to a tool called
check_eligibility that checks whether a patient of a given age (and, if mentioned, sex) meets
a trial's eligibility requirements.

If the question mentions a specific NCT ID and a patient age, use the tool to check eligibility -
don't try to reason about ages or sex requirements yourself.

Otherwise, answer using the trial information below.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

def rag_agentic(query):
    nct_match = re.search(r"NCT\d{8}", query)

    if nct_match:
        messages = [{"role": "user", "content": query}]
    else:
        results = search_trials(query, num_results=5)
        prompt = build_agentic_prompt(query, results)
        messages = [{"role": "user", "content": prompt}]

    response = get_client().chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools,
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = check_eligibility(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })
        final = get_client().chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
        )
        return final.choices[0].message.content

    return msg.content

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask(question: Question):
    answer = rag_agentic(question.query)
    return {"answer": answer}