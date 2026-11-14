import streamlit as st
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
 
load_dotenv()
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
 
st.set_page_config(page_title="Kwest", layout="wide", initial_sidebar_state="expanded")
 
# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { display: none !important; }

[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    padding-bottom: 80px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
    width: 230px !important;
    min-width: 230px !important;
}   
[data-testid="stSidebar"] * { color: #d4d4d4 !important; }
 
/* Sidebar nav buttons — flat list, no circles, Claude-style highlight on active */
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: transparent !important;
    border: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 14px !important;
    padding: 8px 10px !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    width: 100%;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: #1c1c1c !important;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 0.5rem !important;
}

/* "primary" type = active page */
[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
    background: #2a2a2a !important;
    color: #ffffff !important;
    font-weight: 500 !important;
}

[data-testid="stMarkdownContainer"] hr {
    margin: 8px 0 !important;
}
 
/* Tabs — make them look closer to Claude's section switcher */
[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 14px;
    padding: 8px 16px;
}
 
/* Chat */
[data-testid="stChatMessage"] { padding: 10px 0; }
[data-testid="stChatMessageContent"] p {
    font-size: 16px; 
    line-height: 1.75;
    margin: 0 0 6px 0;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 4rem !important;
}

[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 1.5rem;
    left: 260px;
    right: 2rem;
    max-width: 700px;
    margin: 0 auto;
    z-index: 999;
    background: transparent;
    padding: 0;
}

[data-testid="stChatInput"] > div {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 28px !important;
    overflow: hidden;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border-radius: 28px !important;
    padding-left: 20px !important;
    padding-right: 25px !important;
}

[data-testid="stAppViewContainer"] {
    overflow: hidden;
}

/* Code viewer */
.stCodeBlock {
    max-height: 650px;
    overflow-y: auto;
    font-size: 12px;
}

 
/* Clean up status elements */
.stAlert p { font-size: 13px; margin: 0; }
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
    st.markdown("## Kwest")
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
        chat_box = st.container(height=520, border=False)

        with chat_box:
            if not st.session_state.messages:
                st.markdown(
                    "<div style='color:rgba(255,255,255,0.25);text-align:center;"
                    "padding:80px 0;font-size:14px;'>"
                    "Set a competition in Settings, then start chatting."
                    "</div>",
                    unsafe_allow_html=True,
                )
            for msg in st.session_state.messages:
                label = "RugvedBane" if msg["role"] == "user" else "Kwest"
                st.markdown(f"**{label}**")
                st.markdown(msg["content"])
                st.markdown("---")

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
