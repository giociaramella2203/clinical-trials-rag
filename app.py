from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os
import re
import json
import requests
from openai import OpenAI
from minsearch import Index

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

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
index = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global all_trials, index

    conditions = ["rheumatoid arthritis", "breast cancer"]
    all_trials = []
    for cond in conditions:
        studies = fetch_trials(cond)
        all_trials.extend([parse_study(s) for s in studies])

    for t in all_trials:
        t["conditions"] = " ".join(t["conditions"]) if t["conditions"] else ""

    index = Index(
        text_fields=["title", "eligibility_criteria", "conditions", "nct_id"],
        keyword_fields=["nct_id"]
    )
    index.fit(all_trials)

    print(f"Loaded {len(all_trials)} trials at startup.")

    yield

    print("Shutting down.")

app = FastAPI(title="Clinical Trials Eligibility Assistant", lifespan=lifespan)

def parse_age(age_str):
    if not age_str:
        return None
    match = re.search(r"\d+", age_str)
    return int(match.group()) if match else None

def check_eligibility(nct_id, patient_age):
    trial = next((t for t in all_trials if t["nct_id"] == nct_id), None)
    if not trial:
        return {"error": "trial not found"}

    min_age = parse_age(trial["min_age"])
    max_age = parse_age(trial["max_age"])

    eligible = True
    reasons = []
    if min_age is not None and patient_age < min_age:
        eligible = False
        reasons.append(f"patient age {patient_age} is below minimum age {min_age}")
    if max_age is not None and patient_age > max_age:
        eligible = False
        reasons.append(f"patient age {patient_age} is above maximum age {max_age}")

    return {
        "nct_id": nct_id,
        "eligible_by_age": eligible,
        "min_age": min_age,
        "max_age": max_age,
        "reasons": reasons,
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": "Check if a patient of a given age is eligible for a specific clinical trial based on its age requirements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string", "description": "The NCT ID of the trial"},
                    "patient_age": {"type": "integer", "description": "The patient's age in years"},
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
check_eligibility that checks whether a patient of a given age meets a trial's age requirements.

If the question mentions a specific NCT ID and a patient age, use the tool to check eligibility -
don't try to reason about ages yourself.

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
        results = index.search(query, num_results=3)
        prompt = build_agentic_prompt(query, results)
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
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
        final = client.chat.completions.create(
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