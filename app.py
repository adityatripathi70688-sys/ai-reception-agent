from database import init_db, insert_call_log, get_all_logs

# Initialize database on app run
init_db()
import os
import streamlit as st
import pandas as pd

from database import init_db, fetch_all_logs, fetch_filtered_logs, get_db_stats
from ai_agent import run_full_pipeline, transcribe_audio_groq

# Optional: load a local .env file if python-dotenv is installed and a .env
# file exists. This is purely a local-dev convenience -- nothing is required
# here, and no key is ever hardcoded in source. If python-dotenv isn't
# installed, we just skip this silently and fall back to the UI input field.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DB_PATH = None

# Env var names checked as fallbacks for the API key fields below.
# Set these in your shell or a local .env file (never commit .env to git):
#   export GROQ_API_KEY="your-key-here"
#   export ANTHROPIC_API_KEY="your-key-here"
ENV_KEY_NAMES = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# ---------------------------------------------------------------------------
# Page config + one-time DB init
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Operations AI Agent & Triage System",
    page_icon="🛠️",
    layout="wide",
)

init_db()  # idempotent - safe to call on every rerun

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.title("🛠️ Enterprise Operations AI Agent & Triage System")
st.caption(
    "Upload a call transcript (or audio file), run the AI triage agent, "
    "and auto-route it to the right team with a full audit trail in PostgreSQL."
)

# ---------------------------------------------------------------------------
# SIDEBAR: upload, API keys, live DB stats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["groq", "anthropic"],
        help="Choose which API the extraction agent should call.",
    )

    # Pre-fill from environment variable / .env if present, so you don't have
    # to paste the key in every session. The text_input value always wins if
    # the user types something, and the raw key is never displayed or logged.
    env_key = os.environ.get(ENV_KEY_NAMES[provider], "")

    api_key = st.text_input(
        f"{provider.capitalize()} API Key",
        value=env_key,
        type="password",
        help=(
            "Your key is used only for this session and is never written to disk. "
            f"Auto-filled from the {ENV_KEY_NAMES[provider]} environment variable if set."
        ),
    )

    if env_key:
        st.caption(f"✅ Loaded from `{ENV_KEY_NAMES[provider]}` env var.")

    model_override = st.text_input(
        "Model override (optional)",
        placeholder="openai/gpt-oss-120b / claude-sonnet-4-6",
        help="Leave blank to use the sensible default for the selected provider.",
    )

    st.divider()
    st.subheader("🎙️ Audio Transcription (Groq Whisper)")

    # Whisper is Groq-only, so it needs a Groq key regardless of which
    # provider is chosen above for the extraction step. Reuse the key
    # already entered above if the provider is groq, otherwise ask/env-fill
    # separately so Anthropic-extraction users can still transcribe audio.
    if provider == "groq":
        whisper_api_key = api_key
        st.caption("Using the Groq key entered above for transcription too.")
    else:
        whisper_env_key = os.environ.get(ENV_KEY_NAMES["groq"], "")
        whisper_api_key = st.text_input(
            "Groq API Key (for Whisper transcription)",
            value=whisper_env_key,
            type="password",
            help="Whisper speech-to-text is only available via Groq, even if "
            "you're using Anthropic for the extraction step above.",
        )
        if whisper_env_key:
            st.caption(f"✅ Loaded from `{ENV_KEY_NAMES['groq']}` env var.")

    whisper_model = st.selectbox(
        "Whisper model",
        options=["whisper-large-v3-turbo", "whisper-large-v3"],
        help="Turbo is faster/cheaper; whisper-large-v3 is slightly more "
        "accurate on difficult audio (heavy accents, background noise).",
    )

    st.divider()
    st.subheader("📁 Upload Input")

    uploaded_file = st.file_uploader(
        "Upload a transcript (.txt) or audio file",
        type=["txt", "wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "mpeg", "mpga"],
        help="Text files are read directly. Audio files can be transcribed "
        "in-app using Groq Whisper (see button under the audio player).",
    )

    if uploaded_file is not None:
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith(".txt"):
            st.session_state.input_text = uploaded_file.read().decode("utf-8", errors="replace")
            # Clear any stale audio from a previous upload in this session.
            st.session_state.pop("audio_bytes", None)
            st.session_state.pop("audio_name", None)
        else:
            # Audio file: keep bytes for the player + Whisper transcription
            # button in the main panel.
            st.session_state["audio_bytes"] = uploaded_file.read()
            st.session_state["audio_name"] = uploaded_file.name
            st.info(
                "Audio uploaded. Click 'Transcribe with Groq Whisper' next "
                "to the audio player below to convert it to text."
            )

    st.divider()
    st.subheader("📊 Database Stats")

    stats = get_db_stats(DB_PATH)
    st.metric("Total Calls Logged", stats["total_calls"])
    st.metric("Average Risk Score", stats["avg_risk_score"])

    if stats["urgency_breakdown"]:
        st.caption("Urgency breakdown")
        st.bar_chart(pd.Series(stats["urgency_breakdown"], name="count"))

    if stats["department_breakdown"]:
        st.caption("Department breakdown")
        st.dataframe(
            pd.Series(stats["department_breakdown"], name="count").to_frame(),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# MAIN PANEL - TOP: input preview
# ---------------------------------------------------------------------------
st.subheader("1️⃣ Input Preview")

col_input, col_audio = st.columns([2, 1])

with col_input:
    st.session_state.input_text = st.text_area(
        "Call transcript / text input",
        value=st.session_state.input_text,
        height=180,
        placeholder="Paste or edit the customer call transcript here...",
    )

with col_audio:
    if st.session_state.get("audio_bytes"):
        st.caption(f"Audio file: {st.session_state.get('audio_name', 'uploaded audio')}")
        st.audio(st.session_state["audio_bytes"])

        transcribe_clicked = st.button(
            "📝 Transcribe with Groq Whisper", use_container_width=True
        )

        if transcribe_clicked:
            if not whisper_api_key.strip():
                st.error("Enter a Groq API key in the sidebar to transcribe audio.")
            else:
                with st.spinner(f"Transcribing with {whisper_model}..."):
                    try:
                        transcript_text = transcribe_audio_groq(
                            audio_bytes=st.session_state["audio_bytes"],
                            filename=st.session_state["audio_name"],
                            api_key=whisper_api_key,
                            model=whisper_model,
                        )
                        if transcript_text:
                            st.session_state.input_text = transcript_text
                            st.success("Transcription complete -- text field updated below.")
                        else:
                            st.warning("Whisper returned empty text. The audio may be silent or unsupported.")
                    except ValueError as size_err:
                        # File too large for Groq's 25MB limit.
                        st.error(str(size_err))
                    except Exception as exc:  # noqa: BLE001
                        # Network error, bad key, rate limit, unsupported
                        # format, etc. Never crash the app -- surface it.
                        st.error(f"Transcription failed: {exc}")
    else:
        st.caption("No audio file uploaded.")

# ---------------------------------------------------------------------------
# MAIN PANEL - MIDDLE: processing button + metrics
# ---------------------------------------------------------------------------
st.subheader("2️⃣ Run AI Triage Pipeline")

run_col, status_col = st.columns([1, 3])

with run_col:
    run_clicked = st.button("🚀 Process Call", type="primary", use_container_width=True)

if run_clicked:
    if not st.session_state.input_text.strip():
        st.error("Please provide transcript text before running the pipeline.")
    elif not api_key.strip():
        st.error(f"Please enter your {provider.capitalize()} API key in the sidebar.")
    else:
        with st.spinner("Running RAG search, LLM extraction, and SQL logging..."):
            result = run_full_pipeline(
                transcript=st.session_state.input_text,
                provider=provider,
                api_key=api_key,
                model=model_override.strip() or None,
                db_path=DB_PATH,
            )
        st.session_state.last_result = result

# --- Render metrics for the most recent result (persists across reruns) ---
result = st.session_state.last_result

if result:
    if result.get("llm_error"):
        st.warning(
            f"⚠️ LLM call failed ({result['llm_error']}). "
            "Falling back to safe defaults -- this call was still logged for manual review."
        )

    st.markdown("### 📋 Triage Results")

    badge_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    urgency = result.get("urgency", "Medium")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"**Urgency**")
        st.markdown(f"## {badge_colors.get(urgency, '⚪')} {urgency}")

    with m2:
        st.markdown("**Risk Score**")
        risk = result.get("risk_score", 5)
        st.markdown(f"## {risk} / 10")
        st.progress(risk / 10)

    with m3:
        st.markdown("**Sentiment**")
        sentiment_icons = {"Frustrated": "😠", "Neutral": "😐", "Satisfied": "🙂"}
        sentiment = result.get("sentiment", "Neutral")
        st.markdown(f"## {sentiment_icons.get(sentiment, '')} {sentiment}")

    with m4:
        st.markdown("**Assigned Department**")
        st.markdown(f"## 🏷️ {result.get('assigned_department', 'Support')}")

    st.divider()

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("**Caller Name:** " + result.get("caller_name", "Unknown"))
        st.markdown("**Callback Number:** " + result.get("callback_number", "Unknown"))
        st.markdown("**Intent:** " + result.get("intent", "Not specified"))
        st.markdown("**Action Required:** " + result.get("action_required", "Manual review required"))

    with detail_col2:
        st.markdown(f"**RAG Policy Match** (score: {result.get('rag_score', 0)}, backend: `{result.get('rag_backend', 'n/a')}`)")
        st.info(result.get("rag_policy_matched", "None"))
        st.success(result.get("sql_action_status", "Not logged"))

# ---------------------------------------------------------------------------
# MAIN PANEL - BOTTOM: historical call log viewer
# ---------------------------------------------------------------------------
st.divider()
st.subheader("3️⃣ Historical Call Logs")

filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

with filter_col1:
    urgency_filter = st.selectbox("Filter by urgency", ["All", "High", "Medium", "Low"])

with filter_col2:
    all_logs_for_depts = fetch_all_logs(DB_PATH)
    dept_options = ["All"] + sorted(all_logs_for_depts["assigned_department"].dropna().unique().tolist()) \
        if not all_logs_for_depts.empty else ["All"]
    department_filter = st.selectbox("Filter by department", dept_options)

with filter_col3:
    search_text = st.text_input("Search (name, intent, phone number)", placeholder="e.g. cancellation, John, 555-...")

filtered_df = fetch_filtered_logs(
    db_path=DB_PATH,
    urgency=urgency_filter,
    department=department_filter,
    search_text=search_text or None,
)

if filtered_df.empty:
    st.info("No call logs match the current filters yet. Process a call above to get started.")
else:
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score", min_value=1, max_value=10, format="%d"
            ),
            "timestamp": st.column_config.DatetimeColumn("Timestamp"),
        },
    )
    st.caption(f"Showing {len(filtered_df)} of {stats['total_calls']} total logged calls.")
