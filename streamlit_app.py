import html
import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv()

# set_page_config must be the first Streamlit call
_logo = BASE_DIR / "Kwest_logo.svg"
st.set_page_config(
    page_title="Kwest",
    page_icon=str(_logo) if _logo.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Credentials (Streamlit Cloud secrets -> local files) ───────────────────────
def _setup_kaggle_credentials() -> bool:
    """Write ~/.kaggle/kaggle.json from secrets or env. Returns True if creds exist."""
    user, key = None, None
    try:
        user = st.secrets.get("kaggle_username")
        key = st.secrets.get("kaggle_key")
    except Exception:
        pass  # no secrets.toml, fall back to env / existing kaggle.json
    user = user or os.getenv("KAGGLE_USERNAME") or os.getenv("username")
    key = key or os.getenv("KAGGLE_KEY")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if user and key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": user, "key": key}))
        try:
            os.chmod(kaggle_json, 0o600)
        except OSError:
            pass  # chmod is a no-op on Windows
        return True
    return kaggle_json.exists()


def _llm_key_present() -> bool:
    if os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return True
    try:
        return bool(st.secrets.get("rapid_router_key"))
    except Exception:
        return False


KAGGLE_OK = _setup_kaggle_credentials()
LLM_OK = _llm_key_present()


# ── Styles ─────────────────────────────────────────────────────────────────────
LOGO_MARK = (
    '<svg width="{s}" height="{s}" viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg" '
    'aria-hidden="true"><rect x="3" y="4" width="7" height="28" rx="1" fill="{c}"/>'
    '<path d="M13 14 L26 3 L33 8 L20 17 Z" fill="{c}"/>'
    '<path d="M13 20 L20 17 L33 27 L26 33 Z" fill="{c}"/></svg>'
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #F4F7F2;
    --surface: #FFFFFF;
    --forest: #0E3B2A;
    --forest-2: #134A35;
    --green: #1F6B4A;
    --green-hover: #17563B;
    --mint: #CFE5D4;
    --mint-soft: #E4ECE2;
    --leaf: #7BC48F;
    --text: #1B2620;
    --muted: #66746B;
    --border: #D5DED3;
    --amber: #B7791F;
    --red: #B3402F;
    --display: 'Bricolage Grotesque', 'Segoe UI', sans-serif;
    --body: 'Instrument Sans', 'Segoe UI', system-ui, sans-serif;
    --mono: 'JetBrains Mono', Consolas, monospace;
}

/* Fonts: set on text elements only so Material icon glyphs keep their own font */
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
.stApp textarea, .stApp td, .stApp th, .stApp summary {
    font-family: var(--body);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: var(--display);
    letter-spacing: -0.02em;
    color: var(--forest);
}
.stApp code, .stApp pre, .stApp kbd {
    font-family: var(--mono) !important;
}
[data-testid="stIconMaterial"], .material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
}

/* App chrome */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] [data-testid="stMainMenu"],
[data-testid="stToolbar"] [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg); }
[data-testid="stMainBlockContainer"] {
    width: 100%;
    max-width: 1480px;
    padding: 2.2rem clamp(2rem, 5vw, 5rem) 7rem;
    margin-left: auto;
    margin-right: auto;
}
::selection { background: var(--green); color: #fff; }

/* Bottom bar that holds the chat input */
[data-testid="stBottom"], [data-testid="stBottom"] > div {
    background: linear-gradient(to top, var(--bg) 70%, rgba(244,247,242,0)) !important;
}
[data-testid="stBottom"] { padding-bottom: 0 !important; }

/* Sidebar: deep forest is the one bold surface */
[data-testid="stSidebar"] {
    background: var(--forest) !important;
    border-right: none !important;
}
[data-testid="stSidebar"][aria-expanded="true"] {
    width: 248px !important;
    min-width: 248px !important;
}
[data-testid="stSidebar"] * { color: #D7E8DC; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; margin: 1rem 0 !important; }
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] button * { color: #9FC4AC !important; }

.kw-brand { display: flex; align-items: center; gap: 12px; padding: 6px 4px 2px; }
.kw-brand span {
    font-family: var(--display); font-weight: 700; font-size: 28px;
    letter-spacing: -0.04em; color: #F2F8F3 !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%; justify-content: flex-start; text-align: left;
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: #B5CFBF !important; border-radius: 8px !important;
    padding: 0.5rem 0.9rem !important; font-size: 15px; font-weight: 500;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.08) !important; color: #fff !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--leaf) !important; color: var(--forest) !important; font-weight: 600;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] * { color: var(--forest) !important; }
[data-testid="stSidebar"] .stButton > button:focus-visible { outline: 2px solid var(--leaf); outline-offset: 2px; }

.kw-side-label { font-size: 13px; color: #8FB59D !important; margin: 0 0 4px 4px; }
.kw-slug {
    display: block; margin: 0 0 14px 4px; padding: 6px 10px; border-radius: 8px;
    background: rgba(255,255,255,0.07); font-family: var(--mono); font-size: 12.5px;
    color: #E6F2E9 !important; word-break: break-all;
}
.kw-slug.empty { font-family: var(--body); color: #8FB59D !important; font-style: italic; }
.kw-status { display: flex; align-items: center; gap: 8px; margin: 3px 0 3px 4px; font-size: 13.5px; }
.kw-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.kw-dot.ok { background: var(--leaf); box-shadow: 0 0 0 3px rgba(123,196,143,0.2); }
.kw-dot.off { background: #D98A7A; }

/* Page titles */
.kw-title { font-family: var(--display); font-weight: 700; font-size: 2rem; color: var(--forest); letter-spacing: -0.03em; margin: 0 0 0.2rem; }
.kw-sub { color: var(--muted); margin: 0 0 1.4rem; font-size: 15px; }

/* Chat */
.kw-empty {
    text-align: center; padding: 14vh 1rem 0; color: var(--muted);
}
.kw-empty .kw-empty-mark { display: inline-block; margin-bottom: 14px; }
.kw-empty h3 { margin: 0 0 6px; font-size: 1.5rem; }
.kw-empty p { margin: 0; font-size: 15px; }

.kw-user {
    display: flex; justify-content: flex-end;
    width: 100%; margin: 1.4rem 0 1rem;
    padding: 0 1.5rem 0 1rem; box-sizing: border-box;
}
.kw-user .kw-bubble {
    background: var(--green); color: #F4FAF5; padding: 0.7rem 1.05rem;
    border-radius: 18px 18px 5px 18px; max-width: 58%; line-height: 1.6;
    overflow-wrap: anywhere; font-size: 15.5px;
}
/* Keep Kwest's replies slightly inset from the left edge. */
[class*="st-key-kw_assistant_"] {
    width: 96%; max-width: 100%; margin-left: 0.75rem;
}
.kw-who { display: flex; align-items: center; gap: 9px; margin: 1.6rem 0 0.1rem; font-weight: 600; color: var(--forest); font-size: 23px; }
.kw-who-label { text-decoration: underline; text-decoration-color: rgba(14,59,42,0.72); text-decoration-thickness: 1px; text-underline-offset: 4px; text-decoration-skip-ink: auto; }

[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { line-height: 1.65; }

.kw-thinking { display: inline-flex; gap: 5px; padding: 8px 0 4px; }
.kw-thinking i { width: 7px; height: 7px; border-radius: 50%; background: var(--green); opacity: 0.35; animation: kw-pulse 1.2s infinite ease-in-out; }
.kw-thinking i:nth-child(2) { animation-delay: 0.15s; }
.kw-thinking i:nth-child(3) { animation-delay: 0.3s; }
@keyframes kw-pulse { 0%, 80%, 100% { opacity: 0.25; transform: scale(0.85); } 40% { opacity: 1; transform: scale(1.1); } }
@media (prefers-reduced-motion: reduce) { .kw-thinking i { animation: none; opacity: 0.7; } }

/* Chat input */
[data-testid="stChatInput"] {
    width: 76% !important;
    max-width: 1100px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    transform: translateY(32px);
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 22px !important;
    box-shadow: 0 4px 18px rgba(14,59,42,0.07);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 3px rgba(31,107,74,0.16);
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important; caret-color: var(--green);
    min-height: 2.8rem !important;
}
[data-testid="stChatInput"] button { background: var(--green) !important; color: #fff !important; border-radius: 12px !important; }
[data-testid="stChatInput"] button:hover { background: var(--green-hover) !important; }
[data-testid="stChatInput"] button:disabled { background: var(--mint) !important; }
[data-testid="stChatInput"] button svg { fill: #fff !important; color: #fff !important; }

/* Buttons in main area */
.stApp [data-testid="stMain"] .stButton > button[kind="primary"] {
    background: var(--green); color: #fff; border: 1px solid var(--green); border-radius: 10px;
}
.stApp [data-testid="stMain"] .stButton > button[kind="primary"]:hover { background: var(--green-hover); border-color: var(--green-hover); }
.stApp [data-testid="stMain"] .stButton > button[kind="secondary"] {
    background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 10px;
}
.stApp [data-testid="stMain"] .stButton > button[kind="secondary"]:hover { border-color: var(--green); color: var(--green); }

/* Inputs, tables, code, alerts */
[data-testid="stTextInput"] input {
    background: var(--surface) !important; border-radius: 10px !important; border-color: var(--border) !important;
}
[data-testid="stTextInput"] input:focus { border-color: var(--green) !important; box-shadow: 0 0 0 3px rgba(31,107,74,0.16) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
[data-testid="stCode"] { border: 1px solid var(--border); border-radius: 12px; max-height: 640px; overflow: auto; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stMarkdownContainer"] hr { border-color: var(--border) !important; }

/* Experiments dashboard */
.kw-exp-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 1.2rem 0 1.5rem; }
.kw-exp-stat { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(14,59,42,0.035); }
.kw-exp-stat-label { color: var(--muted); font-size: 12.5px; font-weight: 600; letter-spacing: .02em; margin-bottom: 7px; }
.kw-exp-stat-value { color: var(--forest); font-family: var(--display); font-size: 1.65rem; line-height: 1.15; font-weight: 700; }
.kw-exp-stat-foot { color: var(--muted); font-size: 12px; margin-top: 5px; }
.kw-exp-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 1rem; }
.kw-exp-card { min-width: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 17px 18px 15px; box-shadow: 0 2px 10px rgba(14,59,42,0.035); transition: border-color .15s, box-shadow .15s, transform .15s; }
.kw-exp-card:hover { border-color: #A9C9B1; box-shadow: 0 6px 18px rgba(14,59,42,0.07); transform: translateY(-1px); }
.kw-exp-card-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 11px; }
.kw-exp-title { color: var(--forest); font-weight: 600; font-size: 15px; line-height: 1.45; overflow-wrap: anywhere; }
.kw-exp-score { flex: none; color: #14532D; background: #E5F3E8; border: 1px solid #C5E2CC; border-radius: 9px; padding: 5px 9px; font-family: var(--mono); font-size: 13px; font-weight: 600; white-space: nowrap; }
.kw-exp-meta { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 12px; margin-bottom: 12px; }
.kw-exp-notes { border-top: 1px solid #E8EEE7; padding-top: 11px; color: #4E5E53; font-size: 13px; line-height: 1.55; overflow-wrap: anywhere; }
.kw-exp-notes-label { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
.kw-exp-empty { background: var(--surface); border: 1px dashed #C7D5C7; border-radius: 14px; padding: 25px; color: var(--muted); }
@media (max-width: 760px) { .kw-exp-summary { grid-template-columns: 1fr; } .kw-exp-list { grid-template-columns: 1fr; } }



/* Settings and About pages */
.kw-page-intro { color: var(--muted); font-size: 14px; margin: -2px 0 18px; }
.kw-section-kicker { color: var(--green); font-size: 11px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; margin-bottom: 6px; }
.kw-about-hero { background: linear-gradient(135deg, #E8F2E8 0%, #F8FAF6 100%); border: 1px solid #D4E3D3; border-radius: 18px; padding: 25px 27px; margin: 12px 0 20px; }
.kw-about-hero h3 { margin: 0 0 8px; color: var(--forest); font-family: var(--display); font-size: 1.45rem; }
.kw-about-hero p { color: #526258; line-height: 1.65; margin: 0; }
.kw-about-stack { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.kw-about-chip { background: rgba(255,255,255,.78); border: 1px solid #D5E2D3; color: #315A40; border-radius: 999px; padding: 6px 11px; font-size: 12px; font-weight: 600; }
.kw-about-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 13px; margin: 12px 0 22px; }
.kw-about-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 17px 18px; box-shadow: 0 2px 10px rgba(14,59,42,.035); }
.kw-about-card .num { color: var(--green); font-family: var(--mono); font-size: 12px; font-weight: 700; margin-bottom: 8px; }
.kw-about-card h4 { color: var(--forest); font-size: 15px; margin: 0 0 5px; }
.kw-about-card p { color: var(--muted); font-size: 13px; line-height: 1.55; margin: 0; }
.kw-about-command { display: grid; grid-template-columns: minmax(220px,.9fr) minmax(0,1.4fr); gap: 12px; align-items: center; background: var(--surface); border: 1px solid var(--border); border-radius: 11px; padding: 13px 15px; margin: 9px 0; }
.kw-about-command code { color: var(--green); background: #EFF5EF; border-radius: 6px; padding: 5px 7px; overflow-wrap: anywhere; }
.kw-about-command span { color: #526258; font-size: 13px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border) !important; border-radius: 15px !important; background: var(--surface); box-shadow: 0 2px 10px rgba(14,59,42,.035); }
@media (max-width: 760px) { .kw-about-grid { grid-template-columns: 1fr; } .kw-about-command { grid-template-columns: 1fr; gap: 7px; } }

/* Compact kernel status shown inside the chat conversation */
.kw-kernel-inline { display: inline-flex; align-items: center; gap: 7px; margin: 7px 0 3px 26px; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--border); background: rgba(255,255,255,.72); color: var(--muted); font-size: 11.5px; line-height: 1.35; }
.kw-kernel-inline .kw-dot { width: 6px; height: 6px; }
.kw-kernel-inline.success { background: #EAF5EC; border-color: #C9E3CE; color: #25613A; }
.kw-kernel-inline.success .kw-dot { background: #2E9E5B; }
.kw-kernel-inline.warning { background: #FBF4E7; border-color: #EBDDBB; color: #805A18; }
.kw-kernel-inline.warning .kw-dot { background: var(--amber); }
.kw-kernel-inline.error { background: #FBEDEA; border-color: #EDC9C1; color: #8B382B; }
.kw-kernel-inline.error .kw-dot { background: var(--red); }
.kw-kernel-inline.info .kw-dot { background: #8AA395; }

/* Narrow screens */
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
    [data-testid="stChatInput"] { width: 94% !important; }
    [class*="st-key-kw_assistant_"] { width: 96%; margin-left: 0.35rem; }
    .kw-user { padding-right: 0.35rem; }
    .kw-user .kw-bubble { max-width: 90%; }
    .kw-kernel-inline { margin-left: 22px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "messages": [],
        "chat_history": [],
        "current_code": "",
        "kernel_status": "No kernel run yet",
        "kernel_status_type": "info",
        "comp_slug": "",
        "agent": None,
        "page": "Chat",
        "pending_prompt": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_agent(slug: str):
    import agent_tools
    from Kwest import build_agent, create_model

    agent_tools.COMP_SLUG = slug
    model = create_model()
    return build_agent(model, slug)


def _read_solution() -> str:
    slug = st.session_state.comp_slug
    if not slug:
        return ""
    path = BASE_DIR / "solutions" / slug / "solution.py"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _content_as_text(content) -> str:
    """Normalize LangChain message content, including multimodal text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                pieces.append(block["text"])
        return "\n".join(pieces)
    return str(content) if content is not None else ""


def _parse_kernel_status(text: str):
    up = text.upper()
    if any(term in up for term in ("KERNEL STATUS: COMPLETE", "KERNEL COMPLETED", "KERNELWORKERSTATUS.COMPLETE", "STATUS: COMPLETE")):
        return "Complete", "success"
    if any(term in up for term in ("KERNEL TIMEOUT", "KERNELWORKERSTATUS.TIMED_OUT", "STATUS: TIMEOUT")):
        return "Timed out · check Kaggle", "warning"
    if any(term in up for term in ("PUSH FAILED", "KERNEL ERROR", "KERNEL FAILED", "KERNELWORKERSTATUS.ERROR", "STATUS: ERROR")):
        return "Error · check execution logs", "error"
    if any(term in up for term in ("KERNEL CANCELLED", "KERNELWORKERSTATUS.CANCELLED")):
        return "Cancelled", "warning"
    if any(term in up for term in (
        "PUSHED SUCCESSFULLY", "MONITORING EXECUTION", "KERNEL PUSHED",
        "WAITING 30S FOR STATUS", "KERNELWORKERSTATUS.RUNNING",
        "KERNELWORKERSTATUS.QUEUED", "STATUS: RUNNING", "STATUS: QUEUED",
    )):
        return "Pushed · waiting for Kaggle execution", "warning"
    return None, None


def _handle_log(prompt: str) -> str:
    slug = st.session_state.comp_slug
    if not slug:
        return "Set a competition slug in Settings before logging experiments."
    parts = [p.strip() for p in prompt[4:].strip().split("|")]
    desc = parts[0] if len(parts) > 0 else ""
    score = parts[1] if len(parts) > 1 else ""
    notes = parts[2] if len(parts) > 2 else ""
    if not desc:
        return "Format: `log: description | score | notes`"
    from memory import save_experiment

    save_experiment(slug, desc, score, notes)
    st.session_state.agent = _build_agent(slug)
    return (
        f"Experiment logged for **{slug}**\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Score | {score} |\n"
        f"| Description | {desc} |\n"
        f"| Notes | {notes} |"
    )


def _invoke_agent(prompt: str) -> str:
    if st.session_state.agent is None:
        st.session_state.agent = _build_agent(st.session_state.comp_slug)

    # Clear the previous run's badge so a non-kernel answer cannot inherit stale status.
    st.session_state.kernel_status = "No kernel status in this response"
    st.session_state.kernel_status_type = "info"

    result = st.session_state.agent.invoke(
        {"messages": st.session_state.chat_history + [("human", prompt)]}
    )
    result_messages = result.get("messages", [])
    response = _content_as_text(result_messages[-1].content) if result_messages else ""

    # Tool execution details usually live in intermediate ToolMessages, not the final
    # assistant response. Inspect every returned message to find Kaggle's status text.
    execution_trace = "\n".join(
        _content_as_text(getattr(message, "content", ""))
        for message in result_messages
    )
    label, kind = _parse_kernel_status(execution_trace + "\n" + response)
    if label:
        st.session_state.kernel_status = label
        st.session_state.kernel_status_type = kind

    from langchain_core.messages import AIMessage, HumanMessage

    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=response[:400]))
    st.session_state.chat_history = st.session_state.chat_history[-6:]

    new_code = _read_solution()
    if new_code:
        st.session_state.current_code = new_code

    return response


def _dispatch(prompt: str) -> str:
    if prompt.strip().lower().startswith("log:"):
        return _handle_log(prompt.strip())
    if not st.session_state.comp_slug:
        return "Set a competition slug in Settings first."
    return _invoke_agent(prompt)


def _page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="kw-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="kw-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("Chat", "Chat"),
    ("Kwest’s Code", "Kernel"),
    ("Experiments", "Experiments"),
    ("Settings", "Settings"),
    ("About", "About"),
]

with st.sidebar:
    st.markdown(
        f'<div class="kw-brand">{LOGO_MARK.format(s=34, c="#7BC48F")}<span>Kwest</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    for label, page_key in NAV_ITEMS:
        if st.button(
            label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if st.session_state.page == page_key else "secondary",
        ):
            st.session_state.page = page_key
            st.rerun()

    st.markdown("---")

    slug = st.session_state.comp_slug
    st.markdown('<div class="kw-side-label">Competition</div>', unsafe_allow_html=True)
    if slug:
        st.markdown(f'<span class="kw-slug">{html.escape(slug)}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="kw-slug empty">Not set</span>', unsafe_allow_html=True)

    st.markdown('<div class="kw-side-label">Connections</div>', unsafe_allow_html=True)
    for name, ok in (("LLM", LLM_OK), ("Kaggle", KAGGLE_OK)):
        st.markdown(
            f'<div class="kw-status"><span class="kw-dot {"ok" if ok else "off"}"></span>'
            f'{name} {"connected" if ok else "not found"}</div>',
            unsafe_allow_html=True,
        )

page = st.session_state.page


# ── PAGE: Chat ─────────────────────────────────────────────────────────────────
if page == "Chat":
    if not st.session_state.messages and not st.session_state.pending_prompt:
        hint = (
            "Ask it to inspect the data, train a model, or submit."
            if st.session_state.comp_slug
            else "Set a competition slug in Settings to get started."
        )
        st.markdown(
            f'<div class="kw-empty"><div class="kw-empty-mark">{LOGO_MARK.format(s=52, c="#1F6B4A")}</div>'
            f"<h3>What are we building today?</h3><p>{hint}</p></div>",
            unsafe_allow_html=True,
        )

    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            safe = html.escape(msg["content"]).replace("\n", "<br>")
            st.markdown(
                f'<div class="kw-user"><div class="kw-bubble">{safe}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="kw-who">{LOGO_MARK.format(s=24, c="#1F6B4A")}<span class="kw-who-label">Kwest</span></div>',
                unsafe_allow_html=True,
            )
            with st.container(key=f"kw_assistant_{i}"):
                st.markdown(msg["content"])
            status_label = msg.get("kernel_status")
            status_kind = msg.get("kernel_status_type", "info")
            if status_label:
                st.markdown(
                    f'<div class="kw-kernel-inline {status_kind}">'
                    f'<span class="kw-dot"></span>Kernel · {html.escape(status_label)}</div>',
                    unsafe_allow_html=True,
                )

    if st.session_state.pending_prompt:
        st.markdown(
            f'<div class="kw-who">{LOGO_MARK.format(s=24, c="#1F6B4A")}<span class="kw-who-label">Kwest</span></div>'
            '<div class="kw-thinking"><i></i><i></i><i></i></div>',
            unsafe_allow_html=True,
        )

    if prompt := st.chat_input("Message Kwest"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
        st.rerun()

    # second pass: the user message and thinking dots are on screen, now compute the reply
    if st.session_state.pending_prompt:
        pending = st.session_state.pending_prompt
        try:
            reply = _dispatch(pending)
        except Exception as e:
            reply = f"Error: {e}"
        assistant_message = {"role": "assistant", "content": reply}
        if st.session_state.kernel_status != "No kernel status in this response":
            assistant_message["kernel_status"] = st.session_state.kernel_status
            assistant_message["kernel_status_type"] = st.session_state.kernel_status_type
        st.session_state.messages.append(assistant_message)
        st.session_state.pending_prompt = None
        st.rerun()

    # Streamlit reruns the page after both the prompt and the response. Scroll the
    # actual app viewport to the newest message on each render of the chat page.
    # st.iframe accepts trusted HTML strings and supports the parent-document access
    # required for this scroll helper. This replaces deprecated components.html().
    st.iframe(
        """<!doctype html><html><body><script>
        (() => {
          const scrollToLatest = () => {
            try {
              const doc = window.parent.document;
              const main = doc.querySelector('[data-testid="stMain"]');
              const app = doc.querySelector('[data-testid="stAppViewContainer"]');
              const candidates = [main, app, doc.scrollingElement, doc.documentElement, doc.body].filter(Boolean);
              for (const el of candidates) {
                if (el.scrollHeight > el.clientHeight + 8) {
                  el.scrollTo({top: el.scrollHeight, behavior: 'smooth'});
                }
              }
              window.parent.scrollTo({top: doc.documentElement.scrollHeight, behavior: 'smooth'});
            } catch (e) { /* Keep the chat usable if scrolling is unavailable. */ }
          };
          requestAnimationFrame(() => requestAnimationFrame(scrollToLatest));
          setTimeout(scrollToLatest, 180);
        })();
        </script></body></html>""",
        height=1,
        width="stretch",
    )


# ── PAGE: Kernel ───────────────────────────────────────────────────────────────
elif page == "Kernel":
    _page_header("Kwest’s Code", "Generated solution.py for your selected Kaggle competition.")

    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown("#### solution.py")
    with c2:
        if st.button("Refresh", use_container_width=True):
            code = _read_solution()
            if code:
                st.session_state.current_code = code
            else:
                st.toast("No solution.py found.")

    if st.session_state.current_code:
        st.code(st.session_state.current_code, language="python", line_numbers=True)
    else:
        st.caption("No code yet. Ask Kwest to train a model in Chat and it will appear here.")


# ── PAGE: Experiments ──────────────────────────────────────────────────────────
elif page == "Experiments":
    _page_header("Experiments", "A clear history of your model runs, scores and observations.")

    exp_path = BASE_DIR / "experiments.json"
    if not exp_path.exists():
        st.markdown(
            '<div class="kw-exp-empty"><strong>No experiments yet.</strong><br>'
            'In Chat, log a run with <code>log: description | score | notes</code>.</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            all_exps = json.loads(exp_path.read_text(encoding="utf-8"))
            if not isinstance(all_exps, list):
                raise ValueError("experiments.json must contain a list of experiments")
        except (json.JSONDecodeError, OSError, ValueError) as e:
            st.error(f"Could not read experiments.json: {e}")
            all_exps = []

        slug = st.session_state.comp_slug
        if slug:
            exps = [e for e in all_exps if e.get("comp_slug") == slug]
            st.caption(f"Competition  ·  {slug}")
        else:
            exps = all_exps
            st.caption("All competitions")

        def _score_value(value):
            try:
                return float(str(value).strip())
            except (TypeError, ValueError):
                return None

        def _display_value(value):
            if value is None or value == "":
                return "Not recorded"
            if isinstance(value, (list, tuple)):
                return ", ".join(str(x) for x in value)
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)
            return str(value)

        if exps:
            # Sort newest first when dates are available; retain original order otherwise.
            indexed = list(enumerate(exps))
            if any(e.get("date") for e in exps):
                indexed.sort(key=lambda pair: str(pair[1].get("date") or ""), reverse=True)
            ordered = [e for _, e in indexed]

            scored = [(e, _score_value(e.get("score"))) for e in ordered]
            valid_scores = [(e, score) for e, score in scored if score is not None]
            best = max((score for _, score in valid_scores), default=None)
            latest_score = _score_value(ordered[0].get("score"))

            best_text = f"{best:.5f}" if best is not None else "—"
            latest_text = f"{latest_score:.5f}" if latest_score is not None else "—"
            summary_html = f'''<div class="kw-exp-summary">
                <div class="kw-exp-stat"><div class="kw-exp-stat-label">TOTAL EXPERIMENTS</div>
                    <div class="kw-exp-stat-value">{len(ordered)}</div><div class="kw-exp-stat-foot">Logged for this view</div></div>
                <div class="kw-exp-stat"><div class="kw-exp-stat-label">BEST SCORE</div>
                    <div class="kw-exp-stat-value">{html.escape(best_text)}</div><div class="kw-exp-stat-foot">Highest recorded score</div></div>
                <div class="kw-exp-stat"><div class="kw-exp-stat-label">LATEST SCORE</div>
                    <div class="kw-exp-stat-value">{html.escape(latest_text)}</div><div class="kw-exp-stat-foot">Most recently logged run</div></div>
            </div>'''
            st.markdown(summary_html, unsafe_allow_html=True)
            st.markdown("#### Experiment history")

            cards = []
            for i, exp in enumerate(ordered, start=1):
                description = html.escape(_display_value(exp.get("description", "Untitled experiment")))
                score_raw = exp.get("score")
                score_num = _score_value(score_raw)
                score_text = f"{score_num:.5f}" if score_num is not None else _display_value(score_raw)
                date_text = html.escape(_display_value(exp.get("date")))
                notes_text = html.escape(_display_value(exp.get("notes")))
                comp_text = html.escape(_display_value(exp.get("comp_slug")))
                cards.append(f'''<article class="kw-exp-card">
                    <div class="kw-exp-card-top">
                        <div class="kw-exp-title">{description}</div>
                        <div class="kw-exp-score">{html.escape(score_text)}</div>
                    </div>
                    <div class="kw-exp-meta"><span>{date_text}</span><span>·</span><span>{comp_text}</span></div>
                    <div class="kw-exp-notes"><div class="kw-exp-notes-label">Notes</div>{notes_text}</div>
                </article>''')
            st.markdown('<div class="kw-exp-list">' + "".join(cards) + '</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="kw-exp-empty"><strong>No experiments for this competition yet.</strong><br>'
                'Log your next run in Chat using <code>log: description | score | notes</code>.</div>',
                unsafe_allow_html=True,
            )


# ── PAGE: Settings ─────────────────────────────────────────────────────────────
elif page == "Settings":
    _page_header("Settings", "Manage your competition connection and Kwest session.")

    with st.container(border=True):
        st.markdown('<div class="kw-section-kicker">Workspace</div>', unsafe_allow_html=True)
        st.markdown("### Competition")
        st.caption("Choose the Kaggle competition Kwest should work with. Use its competition slug, not the display title.")
        new_slug = st.text_input(
            "Competition slug",
            value=st.session_state.comp_slug,
            placeholder="playground-series-s6e11",
            help="Example: playground-series-s6e11",
        )
        if st.button("Set competition", type="primary"):
            s = new_slug.strip()
            if s:
                st.session_state.comp_slug = s
                st.session_state.agent = None
                st.session_state.kernel_status = "No kernel run yet"
                st.session_state.kernel_status_type = "info"
                import agent_tools

                agent_tools.COMP_SLUG = s
                st.session_state.current_code = _read_solution()
                st.success(f"Competition set to {s}")
            else:
                st.error("Enter a competition slug.")

    st.markdown(" ")
    with st.container(border=True):
        st.markdown('<div class="kw-section-kicker">Conversation</div>', unsafe_allow_html=True)
        st.markdown("### Session management")
        st.caption("Clear the current conversation or reset the app's session state.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Clear chat**")
            st.caption("Removes chat messages and conversation history. Your competition and experiment log remain.")
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.pending_prompt = None
                st.rerun()
        with c2:
            st.markdown("**Reset everything**")
            st.caption("Resets the current Streamlit session, including the selected competition and page state.")
            if st.button("Reset everything", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()


# ── PAGE: About ────────────────────────────────────────────────────────────────
elif page == "About":
    _page_header("About Kwest", "Your workspace for running and tracking Kaggle competition experiments.")

    st.markdown(
        """
        <section class="kw-about-hero">
            <div class="kw-section-kicker">Kaggle competition assistant</div>
            <h3>Experiment, evaluate, improve.</h3>
            <p>Kwest is a personal Kaggle competition agent. It helps inspect competition data, write machine-learning code, run work on Kaggle kernels, submit predictions and keep track of experiments from one chat interface.</p>
            <div class="kw-about-stack">
                <span class="kw-about-chip">LangChain / LangGraph</span>
                <span class="kw-about-chip">Kaggle CLI</span>
                <span class="kw-about-chip">Streamlit</span>
                <span class="kw-about-chip">DeepSeek via Rapid Router</span>
            </div>
        </section>
        <div class="kw-section-kicker">Workflow</div>
        <div class="kw-about-grid">
            <article class="kw-about-card"><div class="num">01 / CONFIGURE</div><h4>Choose a competition</h4><p>Open Settings and enter the Kaggle competition slug, for example <code>playground-series-s6e11</code>.</p></article>
            <article class="kw-about-card"><div class="num">02 / EXPLORE</div><h4>Chat with Kwest</h4><p>Ask Kwest to inspect the data, analyze features, train a model or prepare a submission.</p></article>
            <article class="kw-about-card"><div class="num">03 / EXECUTE</div><h4>Run on Kaggle</h4><p>Review generated <code>solution.py</code> and monitor the latest kernel run from the chat conversation.</p></article>
            <article class="kw-about-card"><div class="num">04 / LEARN</div><h4>Track experiments</h4><p>Log scores and observations, then compare your runs on the Experiments page.</p></article>
        </div>
        <div class="kw-section-kicker">Chat commands</div>
        <div class="kw-about-command"><code>log: description | score | notes</code><span>Save an experiment to your local experiment memory.</span></div>
        <div class="kw-about-command"><code>Anything else</code><span>Kwest interprets your request and decides which available tool to use.</span></div>
        """,
        unsafe_allow_html=True,
    )
