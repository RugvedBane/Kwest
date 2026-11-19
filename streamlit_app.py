import streamlit as st
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
 
load_dotenv()
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": st.secrets["kaggle_username"], "key": st.secrets["kaggle_key"]}, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("rapid_router_key")

st.set_page_config(
    page_title="Kwest",
    page_icon=str(BASE_DIR / "Kwest_logo.svg"),
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --kwest-bg: #E6E9E2;
    --kwest-sidebar: #DCE1D8;
    --kwest-input: #C5CCC1;
    --kwest-surface: #FFFFFF;
    --kwest-green: #174D38;
    --kwest-green-hover: #103D2C;
    --kwest-green-soft: #CCD9CA;
    --kwest-text: #252A26;
    --kwest-muted: #697168;
    --kwest-border: #CDD3C9;
}

/* =================================
   GENERAL LAYOUT
================================= */

#MainMenu,
footer,
[data-testid="stToolbar"] {
    display: none !important;
}


[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stHeader"] [data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    position: fixed !important;
    top: 0.5rem !important;
    left: 0.5rem !important;
    z-index: 999999 !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main {
    background: var(--kwest-bg) !important;
    color: var(--kwest-text) !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 0.5rem !important;
    padding-bottom: 7rem !important;
    min-height: 100vh !important;
}

[data-testid="stAppViewContainer"] {
    overflow: hidden;
}

/* =================================
   GENERAL TEXT
================================= */

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: var(--kwest-text);
}

[data-testid="stMarkdownContainer"] hr {
    border-color: var(--kwest-border) !important;
    margin: 8px 0 !important;
}

[data-testid="stAppViewContainer"] mark,
[data-testid="stAppViewContainer"] .highlight {
    background: #D0DDCE !important;
    color: #173D2D !important;
    padding: 2px 5px;
    border-radius: 4px;
}

::selection {
    background: var(--kwest-green) !important;
    color: #F7F9F5 !important;
}

/* =================================
   SIDEBAR
================================= */

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: var(--kwest-sidebar) !important;
    border-right: 1px solid var(--kwest-border) !important;
    width: 230px !important;
    min-width: 230px !important;
}

[data-testid="stSidebar"] * {
    color: var(--kwest-text);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.5rem !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    align-items: stretch !important;
    gap: 0.5rem !important;
}

/* Sidebar navigation */

[data-testid="stSidebar"] [data-testid="stButton"] {
    width: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button {
    display: flex !important;
    justify-content: flex-start !important;
    text-align: left !important;
    width: 100% !important;
    padding: 9px 12px 9px 16px !important;
    background: transparent !important;
    color: #586158 !important;
    border: none !important;
    border-radius: 7px !important;
    box-shadow: none !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button > div {
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] p {
    width: 100% !important;
    text-align: left !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: #D1D9CE !important;
    color: var(--kwest-text) !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
    background: var(--kwest-green-soft) !important;
    color: var(--kwest-green) !important;
    font-weight: 600 !important;
}

/* Kwest brand */

.kwest-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 4px 16px 4px;
}

.kwest-brand span {
    font-size: 25px;
    font-weight: 750;
    letter-spacing: -1.2px;
    color: var(--kwest-green);
}

/* Competition slug */

.st-emotion-cache-1av66ev {
    background: #D1DBCE !important;
    color: var(--kwest-green) !important;
    border: 1px solid #B7C8B5 !important;
    border-radius: 8px !important;
    padding: 5px 10px !important;
}

.st-emotion-cache-1av66ev * {
    color: var(--kwest-green) !important;
}

/* =================================
   BUTTONS
================================= */

/* Primary buttons */

.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: var(--kwest-green) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--kwest-green) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
    background: var(--kwest-green-hover) !important;
    border-color: var(--kwest-green-hover) !important;
}

/* Secondary buttons */

.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: var(--kwest-text) !important;
    border: 1px solid var(--kwest-border) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: var(--kwest-green) !important;
    color: var(--kwest-green) !important;
}

/* =================================
   TABS
================================= */

[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 14px;
    padding: 8px 16px;
    color: var(--kwest-muted) !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--kwest-green) !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: var(--kwest-green) !important;
}

/* =================================
   CHAT HISTORY
================================= */

.st-key-chat-history {
    height: calc(100vh - 13rem) !important;
    max-height: calc(100vh - 13rem) !important;
    overflow-y: auto !important;
}

.st-key-chat-history [data-testid="stVerticalBlock"] {
    padding-bottom: 1rem !important;
}

/* =================================
   CHAT MESSAGES
================================= */

.user-message {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin: 1.5rem 0 2rem 0;
}

.assistant-message {
    margin: 1.8rem 0 0.4rem 0;
}

.message-label {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--kwest-text);
    margin-bottom: 0.55rem;
}

/* User message bubble */

.user-message .message-content {
    background: var(--kwest-sidebar);
    color: var(--kwest-text);
    padding: 0.8rem 1.1rem;
    border-radius: 16px 16px 4px 16px;
    max-width: 75%;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

/* =================================
   CHAT INPUT
================================= */

[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 1.5rem !important;
    left: 260px;
    right: 2rem;
    max-width: 700px;
    margin: 0 auto;
    z-index: 999 !important;
    background: transparent !important;
    padding: 0;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background: var(--kwest-input) !important;
    border: 1px solid #B6C0B4 !important;
    border-radius: 18px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--kwest-green) !important;
    box-shadow: 0 0 0 1px var(--kwest-green) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--kwest-text) !important;
    caret-color: var(--kwest-green) !important;
    border-radius: 12px !important;
    outline: none !important;
    padding-left: 20px !important;
    padding-right: 25px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #626B61 !important;
    opacity: 1 !important;
}

/* Send button */

[data-testid="stChatInput"] button {
    background: var(--kwest-green) !important;
    color: #F7F9F5 !important;
    border: none !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] button:hover {
    background: var(--kwest-green-hover) !important;
}

[data-testid="stChatInput"] button svg {
    color: #F7F9F5 !important;
    fill: #F7F9F5 !important;
}

[data-testid="stChatInput"] button:focus {
    box-shadow: none !important;
}

/* Remove unwanted input circle */

.st-emotion-cache-1wp6n47 {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

[data-testid="stChatInput"] [class*="1wp6n47"],
[data-testid="stChatInput"] [class*="1wp6n47"]::before,
[data-testid="stChatInput"] [class*="1wp6n47"]::after {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* =================================
   OTHER INPUTS
================================= */

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #FFFFFF !important;
    color: var(--kwest-text) !important;
    border-color: var(--kwest-border) !important;
    border-radius: 8px !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--kwest-green) !important;
    box-shadow: 0 0 0 1px var(--kwest-green) !important;
}

/* =================================
   DATAFRAMES AND CODE
================================= */

[data-testid="stDataFrame"] {
    border: 1px solid var(--kwest-border);
    border-radius: 8px;
    overflow: hidden;
}

.stCodeBlock {
    max-height: 650px;
    overflow-y: auto;
    font-size: 12px;
    border-radius: 8px;
}

/* =================================
   ALERTS AND CAPTIONS
================================= */

.stAlert p {
    font-size: 13px;
    margin: 0;
}

[data-testid="stAlert"][data-baseweb="notification"] {
    background: #D5E2D2 !important;
    color: #173D2D !important;
}

[data-testid="stCaptionContainer"] {
    color: var(--kwest-muted) !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "messages":          [],
        "chat_history":      [],
        "current_code":      "",
        "kernel_status":     "No kernel run yet",
        "kernel_status_type": "info",
        "comp_slug":         "",
        "agent":             None,
        "page":              "Chat",
        "pending_prompt":    None,
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
 
 
def _parse_kernel_status(text: str):
    up = text.upper()
    if "KERNEL STATUS: COMPLETE" in up or "KERNEL COMPLETED" in up:
        return "Complete", "success"
    if "KERNEL TIMEOUT" in up:
        return "Timeout — check Kaggle manually", "warning"
    if "PUSH FAILED" in up or "KERNEL ERROR" in up or "KERNEL FAILED" in up:
        return "Error / Push Failed", "error"
    if "KERNEL CANCELLED" in up:
        return "Cancelled", "warning"
    if "PUSHED SUCCESSFULLY" in up or "MONITORING EXECUTION" in up:
        return "Running on Kaggle...", "warning"
    return None, None
 
 
def _handle_log(prompt: str) -> str:
    slug = st.session_state.comp_slug
    if not slug:
        return "Set a COMP_SLUG in Settings before logging experiments."
    parts = [p.strip() for p in prompt[4:].strip().split("|")]
    desc  = parts[0] if len(parts) > 0 else ""
    score = parts[1] if len(parts) > 1 else ""
    notes = parts[2] if len(parts) > 2 else ""
    if not desc:
        return "Format: log: description | score | notes"
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
 
    result = st.session_state.agent.invoke({
        "messages": st.session_state.chat_history + [("human", prompt)]
    })
    response: str = result["messages"][-1].content
 
    from langchain_core.messages import HumanMessage, AIMessage
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=response[:400]))
    st.session_state.chat_history = st.session_state.chat_history[-6:]
 
    new_code = _read_solution()
    if new_code:
        st.session_state.current_code = new_code
 
    label, kind = _parse_kernel_status(response)
    if label:
        st.session_state.kernel_status      = label
        st.session_state.kernel_status_type = kind
 
    return response
 
 
def _dispatch(prompt: str) -> str:
    if prompt.strip().lower().startswith("log:"):
        return _handle_log(prompt.strip())
    if not st.session_state.comp_slug:
        return "Please set a COMP_SLUG in Settings first."
    return _invoke_agent(prompt)
 
 
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="kwest-brand">
        <svg width="34" height="34" viewBox="0 0 36 36"
            xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="4" width="7" height="28" rx="1" fill="#174D38"/>
            <path d="M13 14 L26 3 L33 8 L20 17 Z" fill="#174D38"/>
            <path d="M13 20 L20 17 L33 27 L26 33 Z" fill="#174D38"/>
        </svg>
        <span>Kwest</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
 
    nav_items = ["Chat", "Experiments", "Settings", "About"]
    for item in nav_items:
        is_active = st.session_state.page == item
        if st.button(
            item,
            key=f"nav_{item}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = item
            st.rerun()
 
    page = st.session_state.page
 
    st.markdown("---")
 
    slug = st.session_state.comp_slug
    st.markdown("**Competition**")
    st.markdown(f"`{slug}`" if slug else "_Not set_")
 
    st.markdown("**API Status**")
    llm_key    = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    kaggle_usr = os.getenv("username", "")
    st.markdown(
        f"{'Connected' if llm_key else 'Not found'} &nbsp; LLM  \n"
        f"{'Connected' if kaggle_usr else 'Not found'} &nbsp; Kaggle",
        unsafe_allow_html=True
    )
 

# ── PAGE: Chat ─────────────────────────────────────────────────────────────────
if page == "Chat":
    tab_chat, tab_kernel = st.tabs(["Chat", "Kernel"])
 
    with tab_chat:
        chat_box = st.container(
        height=720,
        border=False,
        key="chat-history"
        )

        with chat_box:
            if not st.session_state.messages:
                st.markdown(
                    "<div style='color:#929890;text-align:center;"
                    "padding:80px 0;font-size:14px;'>"
                    "Set a competition in Settings, then start chatting."
                    "</div>",
                    unsafe_allow_html=True,
                )
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="user-message">
                            <div class="message-label">You</div>
                            <div class="message-content">{msg["content"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div class="assistant-message">
                            <div class="message-label">Kwest</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(msg["content"])

            # show "thinking" bubble inline, as the last item, while a reply is pending
            if st.session_state.get("pending_prompt"):
                st.markdown("**Kwest**")
                st.markdown("_Thinking..._")

        if prompt := st.chat_input("How can I help you today?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.pending_prompt = prompt
            st.rerun()

    # second pass: actually compute the response now that the bubble has rendered
    if st.session_state.get("pending_prompt"):
        prompt = st.session_state.pending_prompt
        try:
            reply = _dispatch(prompt)
        except Exception as e:
            reply = f"Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.pending_prompt = None
        st.rerun()
 
    with tab_kernel:
        s = st.session_state.kernel_status
        t = st.session_state.kernel_status_type
        {"success": st.success, "error": st.error, "warning": st.warning}.get(t, st.info)(s)
 
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown("#### solution.py")
        with c2:
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            if st.button("Refresh", use_container_width=True):
                code = _read_solution()
                if code:
                    st.session_state.current_code = code
                else:
                    st.toast("No solution.py found.")
 
        if st.session_state.current_code:
            st.code(st.session_state.current_code, language="python", line_numbers=True)
        else:
            st.caption("No code generated yet.")
 
 
# ── PAGE: Experiments ──────────────────────────────────────────────────────────
elif page == "Experiments":
    st.markdown("## Experiments")
 
    exp_path = BASE_DIR / "experiments.json"
    if not exp_path.exists():
        st.info("No experiments logged yet.")
    else:
        with open(exp_path, "r") as f:
            all_exps = json.load(f)
 
        slug = st.session_state.comp_slug
        if slug:
            exps = [e for e in all_exps if e.get("comp_slug") == slug]
            st.caption(f"Filtered to: {slug}  ({len(exps)} experiments)")
        else:
            exps = all_exps
            st.caption(f"All competitions — {len(exps)} experiments total")
 
        if exps:
            import pandas as pd
            df = pd.DataFrame(exps)
            if "date" in df.columns:
                df = df.sort_values("date", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No experiments logged for this competition yet.")
 
 
# ── PAGE: Settings ─────────────────────────────────────────────────────────────
elif page == "Settings":
    st.markdown("## Settings")
 
    st.markdown("#### Competition")
    new_slug = st.text_input(
        "Competition slug",
        value=st.session_state.comp_slug,
        placeholder="playground-series-s6e11",
    )
    if st.button("Set Competition", type="primary"):
        s = new_slug.strip()
        if s:
            st.session_state.comp_slug          = s
            st.session_state.agent              = None
            st.session_state.kernel_status      = "No kernel run yet"
            st.session_state.kernel_status_type = "info"
            import agent_tools
            agent_tools.COMP_SLUG = s
            st.session_state.current_code = _read_solution()
            st.success(f"Competition set to: {s}")
        else:
            st.error("Enter a valid competition slug.")
 
    st.divider()
 
    st.markdown("#### Session")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages     = []
            st.session_state.chat_history = []
            st.rerun()
    with c2:
        if st.button("Reset All", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
 
 
# ── PAGE: About ────────────────────────────────────────────────────────────────
elif page == "About":
    st.markdown("## About Kwest")
    st.markdown("""
Kwest is a personal Kaggle competition agent that writes ML code, runs it on Kaggle kernels,
submits predictions, and tracks experiments — all from a chat interface.
 
**Stack:** LangChain · LangGraph · Kaggle CLI · Streamlit · DeepSeek via Rapid Router
 
**Inspired by:** AutoKaggle (ICLR 2025)
 
---
 
#### How to use
 
1. Go to **Settings** and set your competition slug — e.g. `playground-series-s6e11`
2. Chat with Kwest — ask it to inspect data, train a model, or submit
3. Log experiments with: `log: description | score | notes`
4. View all logged experiments in the **Experiments** page
 
---
 
#### Commands
 
| Command | What it does |
|---|---|
| `log: description \\| score \\| notes` | Save an experiment to memory |
| Anything else | Kwest decides which tool to use |
""")
