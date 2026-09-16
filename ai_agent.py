import json
import re
import requests

from rag_utils import get_rag_engine
from database import insert_call_log

# ---------------------------------------------------------------------------
# LLM provider endpoints
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# NOTE: Groq deprecated llama-3.3-70b-versatile / llama-3.1-8b-instant in
# 2026; openai/gpt-oss-120b is the current recommended default per Groq's
# own migration guidance. Override via the "Model override" field in the
# UI if your account still has access to a different model.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Groq Whisper speech-to-text. whisper-large-v3-turbo is faster/cheaper and
# the recommended default; whisper-large-v3 trades speed for slightly higher
# accuracy on difficult audio. Max file size accepted by Groq is 25MB.
DEFAULT_WHISPER_MODEL = "whisper-large-v3-turbo"
GROQ_WHISPER_MAX_BYTES = 25 * 1024 * 1024

# Safe default payload used whenever extraction or parsing fails for any
# reason. The app must NEVER crash or leave a call untriaged -- worst case,
# it gets logged as Medium/5 for a human to review.
FALLBACK_PAYLOAD = {
    "caller_name": "Unknown",
    "callback_number": "Unknown",
    "intent": "Unable to determine intent (parsing failed)",
    "sentiment": "Neutral",
    "urgency": "Medium",
    "risk_score": 5,
    "assigned_department": "Support",
    "action_required": "Manual review required - automated extraction failed",
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_extraction_prompt(transcript: str, policy_context: str) -> str:
    """
    Build the structured extraction prompt sent to the LLM. Grounds the
    model in retrieved policy context (from Tool 1 / RAG) so its routing
    and risk-scoring decisions reflect actual internal policy rather than
    generic judgment.
    """
    return f"""You are an enterprise customer support triage AI agent. Analyze the
call transcript below and extract structured triage data.

Relevant internal policy context (retrieved via RAG search):
---
{policy_context}
---

Call transcript:
---
{transcript}
---

Extract the following fields and respond with PURE JSON ONLY. Do not include
markdown code fences, explanations, or any text before or after the JSON object.

Required JSON schema:
{{
  "caller_name": "string or 'Unknown' if not stated",
  "callback_number": "string or 'Unknown' if not stated",
  "intent": "concise one-sentence summary of what the caller wants",
  "sentiment": "one of: Frustrated, Neutral, Satisfied",
  "urgency": "one of: High, Medium, Low",
  "risk_score": "integer 1-10, where 10 is highest churn/compliance/safety risk",
  "assigned_department": "one of: Engineering, Billing, Sales, Legal, Support, Retention",
  "action_required": "concise, specific next step for the assigned team, informed by the policy context above"
}}

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------
def call_groq(prompt: str, api_key: str, model: str = DEFAULT_GROQ_MODEL, timeout: int = 30) -> str:
    """
    Call the Groq chat completions endpoint (OpenAI-compatible schema).
    Returns the raw text content of the model's reply.
    Raises requests.RequestException on network/HTTP failure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,  # low temperature: we want consistent, structured output
        "max_tokens": 1000,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_anthropic(prompt: str, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL, timeout: int = 30) -> str:
    """
    Call the Anthropic Messages API. Returns the concatenated text content
    of the model's reply. Raises requests.RequestException on failure.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    # Anthropic responses can contain multiple content blocks; join all text blocks.
    text_blocks = [block["text"] for block in data.get("content", []) if block.get("type") == "text"]
    return "\n".join(text_blocks)


def transcribe_audio_groq(
    audio_bytes: bytes,
    filename: str,
    api_key: str,
    model: str = DEFAULT_WHISPER_MODEL,
    language: str = None,
    timeout: int = 120,
) -> str:
    """
    Transcribe an audio file using Groq's hosted Whisper models.

    This calls Groq's OpenAI-compatible transcription endpoint directly via
    `requests` (multipart/form-data), so no extra SDK is required.

    Args:
        audio_bytes: raw bytes of the uploaded audio file.
        filename: original filename (used for content-type inference by the
                  API; must have one of: flac, mp3, mp4, mpeg, mpga, m4a,
                  ogg, wav, webm).
        api_key: Groq API key (Whisper is Groq-only, so this is always a
                 Groq key regardless of which provider is selected for the
                 LLM extraction step elsewhere in the pipeline).
        model: "whisper-large-v3-turbo" (default, fast/cheap) or
               "whisper-large-v3" (higher accuracy on difficult audio).
        language: optional ISO-639-1 code (e.g. "en") to improve accuracy
                  and latency. Left as None for auto-detection.
        timeout: request timeout in seconds. Long audio can take a while
                 even at Groq's speed, so this is generous by default.

    Returns:
        The transcribed text (str).

    Raises:
        ValueError: if the file exceeds Groq's 25MB limit.
        requests.RequestException: on network/HTTP failure (e.g. bad key,
            rate limit, unsupported format) -- callers should catch this,
            same pattern as call_groq()/call_anthropic().
    """
    if len(audio_bytes) > GROQ_WHISPER_MAX_BYTES:
        raise ValueError(
            f"Audio file is {len(audio_bytes) / 1_048_576:.1f}MB, which exceeds "
            f"Groq's 25MB limit for the transcription endpoint. Split or "
            f"compress the file before uploading."
        )

    headers = {"Authorization": f"Bearer {api_key}"}

    # multipart/form-data: file field + form fields, per Groq's
    # OpenAI-compatible /audio/transcriptions endpoint.
    files = {"file": (filename, audio_bytes)}
    data = {
        "model": model,
        "response_format": "json",
        "temperature": "0",  # deterministic transcription
    }
    if language:
        data["language"] = language

    response = requests.post(
        GROQ_WHISPER_URL, headers=headers, files=files, data=data, timeout=timeout
    )
    response.raise_for_status()
    result = response.json()
    return result.get("text", "").strip()


# ---------------------------------------------------------------------------
# JSON parsing guardrail
# ---------------------------------------------------------------------------
def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers some models add despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_llm_json(raw_text: str) -> dict:
    """
    Strictly parse the LLM's JSON output with guardrails.

    Guardrail strategy:
        1. Strip markdown fences if present.
        2. Attempt json.loads() directly.
        3. If that fails, try to extract the first {...} block via regex
           and parse that (handles stray preamble text some models add).
        4. If all parsing attempts fail, fall back to FALLBACK_PAYLOAD so
           the pipeline never raises and the call still gets logged.

    Every returned dict is guaranteed to contain all required keys, either
    from the model or from the fallback defaults, merged field-by-field.
    """
    cleaned = _strip_markdown_fences(raw_text)

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        # Second attempt: pull out the first balanced-looking {...} block.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        # Total parsing failure -- never crash, always return safe defaults.
        return dict(FALLBACK_PAYLOAD)

    # Merge: any field missing or wrong-typed from the model falls back to default.
    result = dict(FALLBACK_PAYLOAD)
    for key in FALLBACK_PAYLOAD:
        if key in parsed and parsed[key] not in (None, ""):
            result[key] = parsed[key]

    # Extra guardrail on risk_score specifically, since it must be an int.
    try:
        result["risk_score"] = int(result["risk_score"])
    except (TypeError, ValueError):
        result["risk_score"] = FALLBACK_PAYLOAD["risk_score"]
    result["risk_score"] = max(1, min(10, result["risk_score"]))

    # Normalize urgency/sentiment casing defensively (e.g. "high" -> "High").
    if isinstance(result.get("urgency"), str):
        result["urgency"] = result["urgency"].strip().capitalize()
        if result["urgency"] not in ("High", "Medium", "Low"):
            result["urgency"] = "Medium"

    if isinstance(result.get("sentiment"), str):
        result["sentiment"] = result["sentiment"].strip().capitalize()
        if result["sentiment"] not in ("Frustrated", "Neutral", "Satisfied"):
            result["sentiment"] = "Neutral"

    return result


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------
def run_full_pipeline(
    transcript: str,
    provider: str,
    api_key: str,
    model: str = None,
    db_path: str = None,
) -> dict:
    """
    Run the complete agent pipeline end-to-end:
        RAG search -> LLM extraction -> JSON guardrail parsing -> SQL insert

    Args:
        transcript: raw customer text/transcript.
        provider: "groq" or "anthropic".
        api_key: API key for the selected provider.
        model: optional model override; defaults per-provider if omitted.
        db_path: optional database path parameter retained for compatibility.
    Returns:
        dict containing:
            - all extracted triage fields
            - rag_policy_matched: the retrieved policy snippet
            - rag_score: similarity score of that match
            - sql_action_status: confirmation the record was logged
            - db_row_id: primary key of the inserted row
            - llm_error: present only if the LLM call itself failed
    """
    # ---- Tool 1: RAG policy search -----------------------------------
    rag_engine = get_rag_engine()
    rag_result = rag_engine.search(transcript)

    prompt = build_extraction_prompt(transcript, rag_result["matched_chunk"])

    # ---- LLM extraction -------------------------------------------------
    llm_error = None
    raw_output = None
    try:
        if provider == "groq":
            raw_output = call_groq(prompt, api_key, model or DEFAULT_GROQ_MODEL)
        elif provider == "anthropic":
            raw_output = call_anthropic(prompt, api_key, model or DEFAULT_ANTHROPIC_MODEL)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as exc:  # noqa: BLE001 - we intentionally catch broadly here
        # Network errors, auth errors, rate limits, etc. all land here.
        # We never let this crash the app -- fall back to safe defaults
        # and surface the error to the UI for visibility.
        llm_error = str(exc)
        raw_output = None

    # ---- Guardrail JSON parsing ------------------------------------------
    if raw_output is not None:
        extracted = parse_llm_json(raw_output)
    else:
        extracted = dict(FALLBACK_PAYLOAD)

    extracted["raw_transcript"] = transcript
    extracted["rag_policy_matched"] = rag_result["matched_chunk"]

    # ---- Tool 2: SQL logging ----------------------------------------------
    row_id = insert_call_log(extracted, db_path=db_path)

    extracted["sql_action_status"] = f"Logged to PostgreSQL (row id {row_id})"
    extracted["db_row_id"] = row_id
    extracted["rag_score"] = rag_result["score"]
    extracted["rag_backend"] = rag_result["backend"]
    if llm_error:
        extracted["llm_error"] = llm_error

    return extracted
