"""
APMOA — Autonomous Process Mining & Optimization
Professional Interactive Streamlit Control Center

Run from the project root:
    python -m streamlit run dashboard.py

The dashboard executes the real orchestrator as a subprocess so backend
stdout/stderr cannot crash Streamlit's event loop. It then parses the
orchestrator output into an interactive operational / functional view,
including model quality, RAG retrieval, LLM reasoning, adaptation,
process-intelligence and optimization results.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from src.storage.dataset_manager import DatasetManager
from src.storage.experiment_manager import ExperimentManager
from dashboard_extensions import (
    render_datasets_page,
    render_experiments_page,
    render_baseline_vs_apmoa,
    render_multi_dataset_comparison,
    render_what_if,
    render_settings,
    render_rag_knowledge_page,
)


# ============================================================================
# APP CONFIG
# ============================================================================

st.set_page_config(
    page_title="APMOA Control Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
ORCHESTRATOR_MODULE = "src.agents.orchestrator"


# ============================================================================
# SESSION STATE
# ============================================================================

DEFAULT_STATE = {
    "execution": None,
    "last_run_ok": False,
    "run_started": None,
    "run_finished": None,
    "selected_nav": "Overview",
    "selected_dataset_id": None,
    "live_phase_status": {},
    "live_current_phase": None,
    "live_terminal_lines": [],
    "selected_experiment_id": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================================
# GLOBAL CSS
# ============================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #070a10;
    --panel: #0e131d;
    --panel2: #111827;
    --border: rgba(255,255,255,.08);
    --text: #f4f7fb;
    --muted: #8d99aa;
    --blue: #5da9ff;
    --cyan: #59e0d0;
    --amber: #ffc86b;
    --red: #ff7373;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(60,130,255,.10), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(0,220,190,.07), transparent 24%),
        var(--bg);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101520 0%, #0b0f17 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.3rem;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,.10);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: rgba(93,169,255,.55);
    transform: translateY(-1px);
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 34px 38px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,.09);
    background:
        linear-gradient(135deg, rgba(22,31,48,.98), rgba(8,12,19,.98));
    box-shadow: 0 25px 70px rgba(0,0,0,.28);
    margin-bottom: 22px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 360px;
    height: 360px;
    right: -120px;
    top: -150px;
    background: radial-gradient(circle, rgba(70,150,255,.18), transparent 68%);
}

.hero-kicker {
    color: var(--blue);
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .17em;
    text-transform: uppercase;
}

.hero-title {
    color: var(--text);
    font-size: 2.55rem;
    font-weight: 800;
    line-height: 1.08;
    margin-top: 8px;
}

.hero-subtitle {
    color: #a1adbd;
    margin-top: 12px;
    font-size: .98rem;
}

.status-pill {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: .70rem;
    font-weight: 800;
    letter-spacing: .07em;
    border: 1px solid rgba(89,224,208,.25);
    background: rgba(89,224,208,.08);
    color: var(--cyan);
}

.status-pill.warn {
    color: var(--amber);
    border-color: rgba(255,200,107,.25);
    background: rgba(255,200,107,.08);
}

.status-pill.fail {
    color: var(--red);
    border-color: rgba(255,115,115,.25);
    background: rgba(255,115,115,.08);
}

.section-title {
    color: var(--text);
    font-size: 1.25rem;
    font-weight: 800;
    margin: 22px 0 5px 0;
}

.section-caption {
    color: var(--muted);
    font-size: .84rem;
    margin-bottom: 15px;
}

.card {
    background: linear-gradient(145deg, rgba(19,26,38,.98), rgba(10,14,21,.98));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 12px 35px rgba(0,0,0,.18);
    margin-bottom: 14px;
}

.card-title {
    color: #f2f5fa;
    font-weight: 800;
    font-size: .95rem;
}

.card-label {
    color: #7f8b9d;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .10em;
    text-transform: uppercase;
}

.big-value {
    color: var(--text);
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 7px;
}

.small-note {
    color: #778395;
    font-size: .76rem;
    margin-top: 4px;
}

.stage {
    padding: 14px 15px;
    border-radius: 13px;
    border: 1px solid var(--border);
    background: rgba(16,22,32,.92);
    margin-bottom: 9px;
}

.stage-num {
    color: var(--blue);
    font-size: .65rem;
    font-weight: 800;
    letter-spacing: .13em;
}

.stage-name {
    color: #f1f4f8;
    font-weight: 750;
    margin-top: 3px;
    font-size: .88rem;
}

.stage-desc {
    color: #7f8b9d;
    font-size: .70rem;
    line-height: 1.45;
    margin-top: 5px;
}

.stage-done { border-color: rgba(89,224,208,.22); }
.stage-running {
    border-color: rgba(93,169,255,.55);
    box-shadow: 0 0 0 1px rgba(93,169,255,.10), 0 0 24px rgba(93,169,255,.08);
}
.stage-warning { border-color: rgba(255,200,107,.25); }
.stage-fail { border-color: rgba(255,115,115,.28); }
.stage-skipped { opacity: .65; }

.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: .62rem;
    font-weight: 800;
    letter-spacing: .05em;
}

.badge-ok { background: rgba(89,224,208,.10); color: var(--cyan); }
.badge-warn { background: rgba(255,200,107,.10); color: var(--amber); }
.badge-fail { background: rgba(255,115,115,.10); color: var(--red); }
.badge-neutral { background: rgba(255,255,255,.06); color: #a5afbd; }

.decision {
    border-left: 3px solid var(--blue);
    background: rgba(93,169,255,.055);
    border-radius: 10px;
    padding: 16px 18px;
    margin: 10px 0;
}

.decision-title {
    color: #eef4fb;
    font-weight: 800;
    font-size: .90rem;
}

.decision-text {
    color: #a3adbb;
    font-size: .82rem;
    line-height: 1.65;
    margin-top: 7px;
}

.rec {
    background: rgba(17,23,34,.98);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px;
    margin-bottom: 11px;
}

.rec-id {
    color: var(--blue);
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .08em;
}

.rec-title {
    color: #f0f3f7;
    font-weight: 800;
    margin-top: 4px;
}

.rec-meta {
    color: #818d9f;
    font-size: .72rem;
    margin-top: 5px;
}

.rec-body {
    color: #a4adbb;
    font-size: .80rem;
    line-height: 1.6;
    margin-top: 9px;
}

.biz-callout {
    padding: 21px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(30,43,63,.95), rgba(12,18,28,.98));
    border: 1px solid rgba(93,169,255,.18);
}

.biz-heading {
    color: #f5f7fb;
    font-size: 1.12rem;
    font-weight: 800;
}

.biz-body {
    color: #a5afbe;
    font-size: .86rem;
    line-height: 1.72;
    margin-top: 10px;
}

.log {
    background: #04060a;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 14px;
    color: #c0c8d4;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: .72rem;
    white-space: pre-wrap;
    max-height: 620px;
    overflow: auto;
}

.footer {
    color: #5d6878;
    text-align: center;
    font-size: .70rem;
    margin-top: 40px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# HELPERS
# ============================================================================

def safe_text(value: Any, fallback: str = "Not available") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def esc(value: Any) -> str:
    return html.escape(safe_text(value))


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def extract_last(pattern: str, text: str, flags: int = re.I) -> str | None:
    matches = re.findall(pattern, text, flags)
    return matches[-1].strip() if matches else None


def extract_metric(text: str, name: str) -> float | None:
    patterns = [
        rf"\b{re.escape(name)}\s*[:=]\s*([0-9]*\.?[0-9]+)",
        rf"\b{re.escape(name.title())}\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ]
    for p in patterns:
        value = extract_last(p, text)
        if value is not None:
            return safe_float(value)
    return None


def extract_int(text: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        value = extract_last(pattern, text)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


def extract_float(text: str, patterns: list[str]) -> float | None:
    """Extract the first valid floating-point value matching any pattern."""
    for pattern in patterns:
        value = extract_last(pattern, text)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def extract_final_model(text: str) -> str:
    patterns = [
        r"Final selected model:\s*(.+)",
        r"Selected model resolved as:\s*(.+)",
        r"Final model:\s*(.+)",
        r"selected_model['\"]?\s*[:=]\s*['\"]?([^,'\"}\n]+)",
    ]
    for p in patterns:
        value = extract_last(p, text)
        if value:
            return value.strip().strip("'\"")
    return "Not available"


def extract_action(text: str) -> str:
    # Prefer the final adaptation block, then final action.
    matches = re.findall(r"^\s*Action\s*:\s*(.+)$", text, re.I | re.M)
    if matches:
        return matches[-1].strip()
    value = extract_last(r"Final action:\s*(.+)", text)
    return value or "Not available"


def extract_strategy(text: str) -> str:
    matches = re.findall(r"^\s*Strategy\s*:\s*(.+)$", text, re.I | re.M)
    return matches[-1].strip() if matches else "Not available"


def extract_reason(text: str) -> str:
    matches = re.findall(r"^\s*Reason\s*:\s*(.+)$", text, re.I | re.M)
    return matches[-1].strip() if matches else "Not available"


def extract_recommendations(text: str) -> list[dict[str, str]]:
    """
    Parses the recommendation format produced by the optimization agent:

        BOT-001 | HIGH | Category
          Activity : ...
          Action   : ...
    """
    pattern = re.compile(
        r"(?P<id>[A-Z]{2,8}-\d+)\s*\|\s*"
        r"(?P<priority>HIGH|MEDIUM|LOW)\s*\|\s*(?P<category>[^\n]+)\n"
        r"\s*Activity\s*:\s*(?P<activity>[^\n]+)\n"
        r"\s*Action\s*:\s*(?P<action>[^\n]+)",
        re.I,
    )

    results = []
    for m in pattern.finditer(text):
        results.append(
            {
                "id": m.group("id").strip(),
                "priority": m.group("priority").upper().strip(),
                "category": m.group("category").strip(),
                "activity": m.group("activity").strip(),
                "action": m.group("action").strip(),
            }
        )
    return results


def extract_history(text: str) -> list[dict[str, str]]:
    rows = []
    pattern = re.compile(
        r"Iteration\s+(\d+)\s*:\s*(.+)",
        re.I,
    )
    for m in pattern.finditer(text):
        rows.append({"iteration": m.group(1), "model": m.group(2).strip()})
    return rows


def extract_process_counts(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}

    candidates = {
        "cases": [
            r"\bCases?\s*[:=]\s*(\d+)",
            r"\bcase_count\s*[:=]\s*(\d+)",
            r"\bNumber of cases\s*[:=]\s*(\d+)",
        ],
        "events": [
            r"\bEvents?\s*[:=]\s*(\d+)",
            r"\bevent_count\s*[:=]\s*(\d+)",
            r"\bNumber of events\s*[:=]\s*(\d+)",
        ],
        "activities": [
            r"\bActivities?\s*[:=]\s*(\d+)",
            r"\bactivity_count\s*[:=]\s*(\d+)",
        ],
    }

    for key, patterns in candidates.items():
        result[key] = extract_int(text, patterns)

    return result


def extract_experiment_id(text: str) -> str | None:
    """Extract the persistent experiment ID created by the orchestrator."""
    patterns = [
        r"Experiment ID\s*:\s*(EXP-[A-Za-z0-9_-]+)",
        r"""experiment_id['"]?\s*[:=]\s*['"]?(EXP-[A-Za-z0-9_-]+)""",
        r"Experiment storage\s*:\s*[^\n]*?/(EXP-[A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        value = extract_last(pattern, text)
        if value:
            return value.strip().strip("'\"")
    return None


def extract_selection_score(text: str) -> float | None:
    patterns = [
        r"Overall selection score:\s*([0-9]*\.?[0-9]+)",
        r"selection_score['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"Selection score:\s*([0-9]*\.?[0-9]+)",
    ]
    for p in patterns:
        value = extract_last(p, text)
        if value is not None:
            return safe_float(value)
    return None


def parse_execution(stdout: str, stderr: str, return_code: int) -> dict[str, Any]:
    """
    Convert the real APMOA orchestrator console output into structured data
    consumed by the dashboard.

    The orchestrator remains the source of truth. This function only parses
    the subprocess output; it does not make process-mining decisions.
    """
    combined = stdout or ""

    if stderr:
        combined += "\n\n--- STDERR ---\n\n" + stderr

    counts = extract_process_counts(combined)
    recommendations = extract_recommendations(combined)
    history = extract_history(combined)
    experiment_id = extract_experiment_id(combined)

    # ------------------------------------------------------------------
    # Model / evaluation
    # ------------------------------------------------------------------
    final_model = extract_final_model(combined)

    iterations = extract_int(
        combined,
        [
            r"Total iterations:\s*(\d+)",
            r"Total Iterations:\s*(\d+)",
        ],
    ) or len(history)

    fitness = extract_metric(combined, "fitness")
    precision = extract_metric(combined, "precision")
    generalization = extract_metric(combined, "generalization")
    simplicity = extract_metric(combined, "simplicity")
    selection_score = extract_selection_score(combined)

    # ------------------------------------------------------------------
    # Reasoning / adaptation
    # ------------------------------------------------------------------
    action = extract_action(combined)
    strategy = extract_strategy(combined)
    reason = extract_reason(combined)

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------
    rag_enabled = bool(
        re.search(
            r"RAG knowledge base connected to Reasoning Agent",
            combined,
            re.IGNORECASE,
        )
    )

    rag_chunks = extract_int(
        combined,
        [
            r"Knowledge chunks retrieved:\s*(\d+)",
            r"RAG chunks retrieved:\s*(\d+)",
            r'''rag_retrieved_count['\"]?\s*[:=]\s*(\d+)''',
        ],
    )

    rag_sources: list[dict[str, Any]] = []
    rag_pattern = re.compile(
        r"-\s*(?P<source>[^\s(]+)\s*\(score=(?P<score>[0-9]*\.?[0-9]+)\)",
        re.IGNORECASE,
    )

    for match in rag_pattern.finditer(combined):
        source = match.group("source").strip()
        score = safe_float(match.group("score"))
        rag_sources.append({"source": source, "score": score})

    # ------------------------------------------------------------------
    # Groq
    # ------------------------------------------------------------------
    groq_success = bool(
        re.search(r"Groq response received", combined, re.IGNORECASE)
    )

    groq_response_time = extract_float(
        combined,
        [
            r"Groq response received in\s*([0-9]*\.?[0-9]+)\s*seconds",
        ],
    )

    # ------------------------------------------------------------------
    # Process Intelligence
    # ------------------------------------------------------------------
    bottlenecks = extract_int(
        combined,
        [r"Potential bottlenecks\s*:\s*(\d+)"],
    )

    deviations = extract_int(
        combined,
        [r"Potential deviations\s*:\s*(\d+)"],
    )

    rework_activities = extract_int(
        combined,
        [r"Activities with rework\s*:\s*(\d+)"],
    )

    transitions_analyzed = extract_int(
        combined,
        [r"Transitions analyzed\s*:\s*(\d+)"],
    )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    optimization_recommendation_count = extract_int(
        combined,
        [r"Recommendations generated\s*:\s*(\d+)"],
    )

    high_priority = extract_int(
        combined,
        [r"High priority\s*:\s*(\d+)"],
    )

    medium_priority = extract_int(
        combined,
        [r"Medium priority\s*:\s*(\d+)"],
    )

    low_priority = extract_int(
        combined,
        [r"Low priority\s*:\s*(\d+)"],
    )

    return {
        "success": return_code == 0,
        "return_code": return_code,
        "experiment_id": experiment_id,

        # Dataset
        **counts,

        # Model
        "final_model": final_model,
        "iterations": iterations,

        # Evaluation
        "fitness": fitness,
        "precision": precision,
        "generalization": generalization,
        "simplicity": simplicity,
        "selection_score": selection_score,

        # Reasoning / adaptation
        "action": action,
        "strategy": strategy,
        "reason": reason,
        "history": history,

        # RAG
        "rag_enabled": rag_enabled,
        "rag_chunks": rag_chunks,
        "rag_sources": rag_sources,

        # Groq
        "groq_success": groq_success,
        "groq_response_time": groq_response_time,

        # Process Intelligence
        "bottlenecks": bottlenecks,
        "deviations": deviations,
        "rework_activities": rework_activities,
        "transitions_analyzed": transitions_analyzed,

        # Optimization
        "optimization_recommendation_count": optimization_recommendation_count,
        "high_priority": high_priority,
        "medium_priority": medium_priority,
        "low_priority": low_priority,
        "recommendations": recommendations,

        # Raw output is retained for debugging and execution-log display.
        "raw_output": combined,
    }


# ============================================================================
# LIVE EXECUTION / PHASE TRACKING
# ============================================================================

# These stages mirror the actual current orchestrator. The orchestrator has
# explicit PHASE 1..6 markers, followed by Process Intelligence and
# Optimization after the autonomous loop.
PIPELINE = [
    ("01", "Data Analysis", "Understand event-log structure and data quality."),
    ("02", "Process Discovery", "Generate candidate process models."),
    ("03", "Model Evaluation", "Measure model quality."),
    ("04", "Model Selection", "Select the strongest candidate."),
    ("05", "Groq Reasoning", "Interpret model behavior and weaknesses."),
    ("06", "Autonomous Adaptation", "Decide whether to keep or rediscover."),
    ("07", "Process Intelligence", "Find bottlenecks, deviations and rework."),
    ("08", "Optimization", "Convert findings into actions."),
]

PHASE_MARKERS = [
    ("01", "PHASE 1 - DATA ANALYSIS"),
    ("02", "PHASE 2 - PROCESS DISCOVERY"),
    ("03", "PHASE 3 - MODEL EVALUATION"),
    ("04", "PHASE 4 - MODEL SELECTION"),
    ("05", "PHASE 5 - GROQ REASONING"),
    ("06", "PHASE 6 - ADAPTATION"),
]


def initial_phase_status() -> dict[str, str]:
    return {number: "READY" for number, _, _ in PIPELINE}


def phase_from_line(line: str) -> str | None:
    '''Map real orchestrator console output to a UI phase number.'''
    upper = line.upper()

    for number, marker in PHASE_MARKERS:
        if marker in upper:
            return number

    # Post-loop stages do not currently have numeric PHASE headings.
    if "PROCESS INTELLIGENCE" in upper:
        return "07"

    if "RUNNING OPTIMIZATION ANALYSIS" in upper or "OPTIMIZATION ANALYSIS" in upper:
        return "08"

    return None


def render_live_pipeline(container, statuses: dict[str, str], current: str | None = None):
    '''Render only high-level phase state; never dump backend logs into UI.'''
    with container.container():
        st.markdown(
            '<div style="color:#f4f7fb;font-size:.78rem;font-weight:800;letter-spacing:.12em;margin-bottom:10px;">LIVE PIPELINE</div>',
            unsafe_allow_html=True,
        )

        for number, name, description in PIPELINE:
            status = statuses.get(number, "READY")
            if status == "RUNNING":
                cls = "stage-running"
                badge_cls = "badge-neutral"
                badge_text = "● RUNNING"
            elif status == "DONE":
                cls = "stage-done"
                badge_cls = "badge-ok"
                badge_text = "✓ COMPLETE"
            elif status == "FAILED":
                cls = "stage-fail"
                badge_cls = "badge-fail"
                badge_text = "✕ FAILED"
            elif status == "SKIPPED":
                cls = "stage-skipped"
                badge_cls = "badge-neutral"
                badge_text = "— SKIPPED"
            else:
                cls = ""
                badge_cls = "badge-neutral"
                badge_text = "○ READY"

            st.markdown(
                f'''
                <div class="stage {cls}">
                    <div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start;">
                        <div>
                            <div class="stage-num">{number}</div>
                            <div class="stage-name">{esc(name)}</div>
                            <div class="stage-desc">{esc(description)}</div>
                        </div>
                        <span class="badge {badge_cls}" style="white-space:nowrap;margin-top:1px;">{badge_text}</span>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

        if current:
            current_name = next(
                (name for number, name, _ in PIPELINE if number == current),
                current,
            )
            st.markdown(
                f'<div style="color:#7f8b9d;font-size:.68rem;margin-top:8px;">Current stage: <span style="color:#5da9ff;font-weight:800;">{esc(current_name)}</span></div>',
                unsafe_allow_html=True,
            )


def run_apmoa(log_path: str | None = None, live_container=None) -> dict[str, Any]:
    '''
    Execute the real orchestrator as a child process and stream its output.

    The same backend output is:
      1. printed line-by-line to the terminal that launched Streamlit;
      2. retained for the final Execution Log tab.

    Streamlit itself receives only phase-level status while the workflow runs.
    '''
    command = [sys.executable, "-m", ORCHESTRATOR_MODULE]

    if log_path:
        command.extend(["--log-path", str(log_path)])
    started = datetime.now()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    statuses = initial_phase_status()
    current_phase: str | None = None
    output_lines: list[str] = []

    st.session_state.live_phase_status = statuses.copy()
    st.session_state.live_current_phase = None
    st.session_state.live_terminal_lines = []

    if live_container is not None:
        render_live_pipeline(live_container, statuses, None)

    try:
        # Popen is required here. subprocess.run(capture_output=True) waits for
        # the entire workflow and therefore hides live terminal output.
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None

        for raw_line in iter(process.stdout.readline, ""):
            line = raw_line.rstrip("\n")
            output_lines.append(line)

            # EXACT backend line -> terminal.
            print(raw_line, end="", flush=True)

            detected = phase_from_line(line)
            if detected:
                if current_phase and current_phase != detected:
                    if statuses.get(current_phase) == "RUNNING":
                        statuses[current_phase] = "DONE"

                current_phase = detected
                statuses[detected] = "RUNNING"

                st.session_state.live_phase_status = statuses.copy()
                st.session_state.live_current_phase = current_phase
                st.session_state.live_terminal_lines = output_lines[-80:]

                if live_container is not None:
                    render_live_pipeline(live_container, statuses, current_phase)

            elif live_container is not None and len(output_lines) % 12 == 0:
                # Keep the active card visibly alive during long-running phases.
                st.session_state.live_terminal_lines = output_lines[-80:]
                render_live_pipeline(live_container, statuses, current_phase)

        return_code = process.wait()

        if current_phase and statuses.get(current_phase) == "RUNNING":
            statuses[current_phase] = "DONE" if return_code == 0 else "FAILED"

        if return_code == 0:
            # If the current orchestrator did not emit an explicit marker for
            # a post-loop stage, do not pretend it ran. Keep such stages READY.
            pass
        else:
            if current_phase:
                statuses[current_phase] = "FAILED"

        combined = "\n".join(output_lines)
        finished = datetime.now()

        # The orchestrator saves structured experiment artifacts. The
        # dashboard owns the subprocess stdout, so persist the exact trace
        # into the same experiment directory.
        experiment_id = extract_experiment_id(combined)
        if experiment_id:
            try:
                ExperimentManager().save_execution_log(
                    experiment_id,
                    combined,
                )
            except Exception as storage_exc:
                print(
                    f"[Dashboard] Could not persist execution.log: {storage_exc}",
                    file=sys.stderr,
                    flush=True,
                )

        st.session_state.live_phase_status = statuses.copy()
        st.session_state.live_current_phase = current_phase
        st.session_state.live_terminal_lines = output_lines[-80:]

        if live_container is not None:
            render_live_pipeline(live_container, statuses, current_phase)

        return {
            "success": return_code == 0,
            "return_code": return_code,
            "stdout": combined,
            "stderr": "",
            "combined": combined,
            "parsed": parse_execution(combined, "", return_code),
            "started_at": started,
            "finished_at": finished,
            "duration": (finished - started).total_seconds(),
            "phase_status": statuses.copy(),
        }

    except Exception as exc:
        finished = datetime.now()
        if current_phase:
            statuses[current_phase] = "FAILED"

        error_text = repr(exc)
        print(error_text, file=sys.stderr, flush=True)

        st.session_state.live_phase_status = statuses.copy()
        st.session_state.live_current_phase = current_phase

        if live_container is not None:
            render_live_pipeline(live_container, statuses, current_phase)

        combined = "\n".join(output_lines)
        if combined:
            combined += "\n\n--- DASHBOARD EXECUTION ERROR ---\n"
        combined += error_text

        return {
            "success": False,
            "return_code": -1,
            "stdout": combined,
            "stderr": error_text,
            "combined": combined,
            "parsed": parse_execution(combined, error_text, -1),
            "started_at": started,
            "finished_at": finished,
            "duration": (finished - started).total_seconds(),
            "phase_status": statuses.copy(),
        }


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:5px 0 18px 0;">
            <div style="color:#5da9ff;font-size:.68rem;font-weight:800;letter-spacing:.16em;">
                APMOA
            </div>
            <div style="color:#f4f7fb;font-size:1.35rem;font-weight:800;margin-top:4px;">
                Control Center
            </div>
            <div style="color:#788496;font-size:.72rem;margin-top:4px;">
                Autonomous Process Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # DATASET SELECTION FOR EXECUTION
    # ------------------------------------------------------------------
    # The Datasets page manages storage. This selector determines which
    # registered dataset is actually passed to the APMOA backend.
    try:
        dataset_manager = DatasetManager()
        registered_datasets = dataset_manager.list_datasets()
    except Exception as exc:
        registered_datasets = []
        st.error(f"Dataset registry unavailable: {exc}")

    if registered_datasets:
        dataset_options = {
            str(item.get("id")): item
            for item in registered_datasets
            if item.get("id")
        }
        ordered_ids = list(dataset_options.keys())

        # Prefer the small mock dataset on a fresh dashboard so the first
        # integration test cannot accidentally launch an expensive BPI run.
        if st.session_state.selected_dataset_id not in ordered_ids:
            if "mock_event_log" in ordered_ids:
                st.session_state.selected_dataset_id = "mock_event_log"
            else:
                st.session_state.selected_dataset_id = ordered_ids[0]

        selected_id = st.selectbox(
            "EXECUTION DATASET",
            options=ordered_ids,
            index=ordered_ids.index(st.session_state.selected_dataset_id),
            format_func=lambda dataset_id: dataset_options[dataset_id].get(
                "name", dataset_id
            ),
            help="This dataset is passed to the APMOA backend when Run APMOA is clicked.",
        )
        st.session_state.selected_dataset_id = selected_id

        selected_dataset = dataset_options[selected_id]
        selected_dataset_path = dataset_manager.get_dataset_path(selected_id)

        st.caption(
            f"{selected_dataset.get('events', 0):,} events · "
            f"{selected_dataset.get('cases', 0):,} cases · "
            f"{selected_dataset.get('activities', 0)} activities"
        )
        st.caption(f"Path: {selected_dataset_path}")
    else:
        selected_dataset = None
        selected_dataset_path = None
        st.warning("No validated dataset is registered.")

    run_clicked = st.button("▶  RUN APMOA", use_container_width=True, type="primary")

    if run_clicked:
        if not selected_dataset_path:
            st.error("Select a validated dataset before running APMOA.")
            st.stop()

        st.session_state.run_started = datetime.now()
        st.session_state.execution = None
        st.session_state.last_run_ok = False
        st.session_state.live_phase_status = initial_phase_status()
        st.session_state.live_current_phase = None

        # This placeholder is refreshed while the real orchestrator is running.
        live_pipeline_placeholder = st.empty()

        with st.spinner("APMOA is executing the autonomous pipeline..."):
            execution = run_apmoa(selected_dataset_path, live_pipeline_placeholder)

        st.session_state.execution = execution
        st.session_state.last_run_ok = execution["success"]
        st.session_state.run_finished = datetime.now()
        st.session_state.selected_experiment_id = execution.get(
            "parsed", {}
        ).get("experiment_id")
        st.rerun()

    if st.button("↻  CLEAR RESULTS", use_container_width=True):
        st.session_state.execution = None
        st.session_state.last_run_ok = False
        st.session_state.run_started = None
        st.session_state.run_finished = None
        st.rerun()

    st.markdown("---")
    st.markdown("### NAVIGATION")

    nav_options = [
        "Overview",
        "Datasets",
        "Experiments",
        "Process Discovery",
        "Model Evaluation",
        "RAG Knowledge",
        "AI Reasoning",
        "Autonomous Adaptation",
        "Process Intelligence",
        "Optimization",
        "Business View",
        "Baseline vs APMOA",
        "Multi-Dataset Comparison",
        "What-If Analysis",
        "Settings",
        "Execution Log",
    ]

    st.session_state.selected_nav = st.radio(
        "Navigate",
        nav_options,
        index=nav_options.index(st.session_state.selected_nav),
        label_visibility="collapsed",
    )

    st.markdown("---")

    live_status = st.session_state.get("live_phase_status") or initial_phase_status()
    live_current = st.session_state.get("live_current_phase")
    render_live_pipeline(st, live_status, live_current)


# ============================================================================
# HERO
# ============================================================================

execution = st.session_state.execution
parsed = execution.get("parsed", {}) if execution else {}

if execution is None:
    hero_status = "ENGINE READY"
    hero_class = ""
elif execution.get("success"):
    hero_status = "RUN COMPLETE"
    hero_class = ""
else:
    hero_status = "RUN COMPLETED WITH ERRORS"
    hero_class = " fail"

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">Autonomous Process Intelligence Platform</div>
        <div class="hero-title">⚙️ APMOA Control Center</div>
        <div class="hero-subtitle">
            Autonomous Process Mining & Optimization
            &nbsp;•&nbsp;
            Discover → Evaluate → Reason → Adapt → Analyze → Optimize
        </div>
        <span class="status-pill{hero_class}">● {hero_status}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# EMPTY STATE
# ============================================================================
# Dataset and RAG management must remain available before the first APMOA run.
# Therefore the empty-state landing panel is shown only on Overview.

if execution is None and st.session_state.selected_nav == "Overview":
    st.markdown(
        """
        <div class="biz-callout">
            <div class="biz-heading">Autonomous process intelligence, from evidence to decision.</div>
            <div class="biz-body">
                Run the APMOA engine to populate model quality, autonomous decisions,
                process risks, optimization recommendations and the functional/business
                interpretation of the run. Datasets and RAG knowledge can be managed
                before execution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">What the platform evaluates</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">The dashboard is designed around the full APMOA operating lifecycle.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("01", "DISCOVER", "Generate and compare process models from the event log."),
        ("02", "EVALUATE", "Measure fitness, precision, generalization and simplicity."),
        ("03", "REASON + ADAPT", "Use autonomous reasoning to determine the next model action."),
        ("04", "ANALYZE + OPTIMIZE", "Turn bottlenecks and deviations into operational actions."),
    ]
    for col, (num, title, body) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">{num} / {title}</div>
                    <div class="decision-text">{esc(body)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# COMMON DATA
# ============================================================================

final_model = parsed.get("final_model", "Not available")
iterations = parsed.get("iterations") or 0
fitness = parsed.get("fitness")
precision = parsed.get("precision")
generalization = parsed.get("generalization")
simplicity = parsed.get("simplicity")
selection_score = parsed.get("selection_score")
action = parsed.get("action", "Not available")
strategy = parsed.get("strategy", "Not available")
reason = parsed.get("reason", "Not available")
recommendations = parsed.get("recommendations", [])
history = parsed.get("history", [])

cases = parsed.get("cases")
events = parsed.get("events")
activities = parsed.get("activities")

# RAG / LLM
rag_enabled = bool(parsed.get("rag_enabled", False))
rag_chunks = parsed.get("rag_chunks")
rag_sources = parsed.get("rag_sources", [])
groq_success = bool(parsed.get("groq_success", False))
groq_response_time = parsed.get("groq_response_time")

# Process Intelligence / Optimization
bottlenecks = parsed.get("bottlenecks")
deviations = parsed.get("deviations")
rework_activities = parsed.get("rework_activities")
transitions_analyzed = parsed.get("transitions_analyzed")
optimization_recommendation_count = parsed.get(
    "optimization_recommendation_count"
)
high_priority = parsed.get("high_priority")
medium_priority = parsed.get("medium_priority")
low_priority = parsed.get("low_priority")


def metric_card(label: str, value: str, note: str) -> str:
    return f"""
    <div class="card">
        <div class="card-label">{esc(label)}</div>
        <div class="big-value">{esc(value)}</div>
        <div class="small-note">{esc(note)}</div>
    </div>
    """


# ============================================================================
# PERSISTENT EXPERIMENTS
# ============================================================================

def render_persistent_experiments_page() -> None:
    """Display experiments persisted by ExperimentManager."""

    st.markdown(
        '<div class="section-title">Persistent Experiments</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">'
        'Every APMOA execution is persisted and can be reopened without '
        'rerunning the process-mining pipeline.'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        manager = ExperimentManager()
        experiments = manager.list_experiments()
    except Exception as exc:
        st.error(f"Experiment storage unavailable: {exc}")
        return

    if not experiments:
        st.info("No persisted APMOA experiments are available yet.")
        return

    experiment_ids = [
        item["experiment_id"]
        for item in experiments
        if item.get("experiment_id")
    ]

    current_id = st.session_state.get("selected_experiment_id")
    if current_id not in experiment_ids:
        current_id = experiment_ids[0]

    selected_id = st.selectbox(
        "EXPERIMENT",
        experiment_ids,
        index=experiment_ids.index(current_id),
        key="persistent_experiment_selector",
    )
    st.session_state.selected_experiment_id = selected_id

    try:
        experiment = manager.get_experiment(selected_id)
    except Exception as exc:
        st.error(f"Could not load {selected_id}: {exc}")
        return

    metadata = experiment.get("metadata", {})
    final_result = experiment.get("final_result", {})
    evaluations = experiment.get("evaluation_results", {})
    reasoning_result = experiment.get("reasoning", {})
    adaptation_result = experiment.get("adaptation", {})
    process_intelligence_result = experiment.get("process_intelligence", {})
    optimization_result = experiment.get("optimization", {})

    summary_cols = st.columns(6)
    selection_value = metadata.get("selection_score")
    if selection_value is None:
        selection_value = final_result.get("selection_score")

    summary_values = [
        (
            "MODEL",
            metadata.get("final_model")
            or final_result.get("selected_model")
            or "—",
        ),
        (
            "ACTION",
            metadata.get("final_action")
            or final_result.get("action")
            or "—",
        ),
        ("ITERATIONS", metadata.get("iterations", "—")),
        (
            "SELECTION",
            f"{selection_value:.4f}"
            if isinstance(selection_value, (int, float))
            else "—",
        ),
        (
            "FITNESS",
            f'{metadata["fitness"]:.4f}'
            if isinstance(metadata.get("fitness"), (int, float))
            else "—",
        ),
        (
            "PRECISION",
            f'{metadata["precision"]:.4f}'
            if isinstance(metadata.get("precision"), (int, float))
            else "—",
        ),
    ]

    for col, (label, value) in zip(summary_cols, summary_values):
        with col:
            st.markdown(
                metric_card(
                    label,
                    str(value),
                    "Persisted experiment data",
                ),
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Experiment Metadata</div>',
        unsafe_allow_html=True,
    )
    st.json(metadata)

    experiment_path = manager.get_experiment_path(selected_id)

    st.markdown(
        '<div class="section-title">Stored Artifacts</div>',
        unsafe_allow_html=True,
    )

    artifact_names = [
        "metadata.json",
        "final_result.json",
        "evaluation_results.json",
        "reasoning.json",
        "adaptation.json",
        "process_intelligence.json",
        "optimization.json",
        "execution.log",
    ]

    artifact_df = pd.DataFrame(
        [
            {
                "Artifact": name,
                "Available": (experiment_path / name).exists(),
            }
            for name in artifact_names
        ]
    )

    st.dataframe(
        artifact_df,
        use_container_width=True,
        hide_index=True,
    )

    tabs = st.tabs(
        [
            "Final Result",
            "Evaluation",
            "Reasoning",
            "Adaptation",
            "Process Intelligence",
            "Optimization",
            "Execution Log",
        ]
    )

    with tabs[0]:
        st.json(final_result) if final_result else st.info(
            "No final-result artifact was stored."
        )

    with tabs[1]:
        st.json(evaluations) if evaluations else st.info(
            "No evaluation artifact was stored."
        )

    with tabs[2]:
        st.json(reasoning_result) if reasoning_result else st.info(
            "No reasoning artifact was stored."
        )

    with tabs[3]:
        st.json(adaptation_result) if adaptation_result else st.info(
            "No adaptation artifact was stored."
        )

    with tabs[4]:
        st.json(process_intelligence_result) if process_intelligence_result else st.info(
            "No process-intelligence artifact was stored."
        )

    with tabs[5]:
        st.json(optimization_result) if optimization_result else st.info(
            "No optimization artifact was stored."
        )

    with tabs[6]:
        log_text = experiment.get("execution_log")
        if log_text:
            st.download_button(
                "Download execution.log",
                data=log_text,
                file_name=f"{selected_id}_execution.log",
                mime="text/plain",
                use_container_width=True,
            )
            st.markdown(
                f'<div class="log">{esc(log_text)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No execution.log has been stored for this experiment.")

    st.caption(f"Storage path: {experiment_path}")


# ============================================================================
# COMMAND CENTER
# ============================================================================

if st.session_state.selected_nav == "Overview":

    st.markdown('<div class="section-title">Executive Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">A single-page view of the current autonomous process-mining run.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    values = [
        ("FINAL MODEL", final_model, "Selected candidate"),
        ("ITERATIONS", str(iterations), "Autonomous cycles"),
        ("FITNESS", f"{fitness:.4f}" if fitness is not None else "—", "Replay quality"),
        ("PRECISION", f"{precision:.4f}" if precision is not None else "—", "Behavioral precision"),
        ("GENERALIZATION", f"{generalization:.4f}" if generalization is not None else "—", "Model robustness"),
        ("SIMPLICITY", f"{simplicity:.4f}" if simplicity is not None else "—", "Structural simplicity"),
    ]

    for col, (label, value, note) in zip(cols, values):
        with col:
            st.markdown(metric_card(label, value, note), unsafe_allow_html=True)

    left, right = st.columns([1.15, .85])

    with left:
        st.markdown('<div class="section-title">Current Autonomous State</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="decision">
                <div class="decision-title">Decision: {esc(action)}</div>
                <div class="decision-text">
                    <b>Strategy:</b> {esc(strategy)}<br>
                    <b>Rationale:</b> {esc(reason)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if selection_score is not None:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">MODEL SELECTION SCORE</div>
                    <div class="big-value">{selection_score:.4f}</div>
                    <div class="small-note">
                        Weighted selection score across the model-quality dimensions.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-title">Process Scope</div>', unsafe_allow_html=True)
        scope_cols = st.columns(3)
        for col, label, value in [
            (scope_cols[0], "CASES", cases),
            (scope_cols[1], "EVENTS", events),
            (scope_cols[2], "ACTIVITIES", activities),
        ]:
            with col:
                st.markdown(
                    metric_card(label, str(value) if value is not None else "—", "From execution output"),
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-title">AI + Knowledge Context</div>', unsafe_allow_html=True)

    rag_overview_cols = st.columns(3)

    with rag_overview_cols[0]:
        st.markdown(
            metric_card(
                "RAG",
                "CONNECTED" if rag_enabled else "NOT CONNECTED",
                "Knowledge base status during reasoning.",
            ),
            unsafe_allow_html=True,
        )

    with rag_overview_cols[1]:
        st.markdown(
            metric_card(
                "RAG CHUNKS",
                str(rag_chunks) if rag_chunks is not None else "—",
                "Retrieved knowledge chunks.",
            ),
            unsafe_allow_html=True,
        )

    with rag_overview_cols[2]:
        st.markdown(
            metric_card(
                "GROQ",
                "SUCCESS" if groq_success else "NOT CONFIRMED",
                (
                    f"{groq_response_time:.2f}s response"
                    if groq_response_time is not None
                    else "LLM response status"
                ),
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Priority Findings</div>', unsafe_allow_html=True)

    if recommendations:
        high = sum(r["priority"] == "HIGH" for r in recommendations)
        medium = sum(r["priority"] == "MEDIUM" for r in recommendations)
        low = sum(r["priority"] == "LOW" for r in recommendations)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, note in [
            (c1, "HIGH", high, "Immediate attention"),
            (c2, "MEDIUM", medium, "Needs attention"),
            (c3, "LOW", low, "Improvement opportunities"),
            (c4, "TOTAL", len(recommendations), "Optimization findings"),
        ]:
            with col:
                st.markdown(metric_card(label, str(value), note), unsafe_allow_html=True)
    else:
        st.info("No structured optimization recommendations were detected.")

    st.markdown('<div class="section-title">Functional Takeaway</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="biz-callout">
            <div class="biz-heading">
                {esc(final_model)} is the selected process model.
            </div>
            <div class="biz-body">
                APMOA completed {iterations} autonomous iteration(s). The model's
                measured fitness is {f"{fitness:.4f}" if fitness is not None else "not available"},
                while precision is {f"{precision:.4f}" if precision is not None else "not available"}.
                The autonomous engine's final decision is <b>{esc(action)}</b>.
                {("The optimization layer produced " + str(len(recommendations)) + " actionable finding(s).")
                 if recommendations else
                 "No structured optimization findings were available in the captured output."}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# PROCESS FLOW
# ============================================================================

elif st.session_state.selected_nav == "Process Discovery":

    st.markdown('<div class="section-title">End-to-End Process Execution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Inspect what each APMOA stage is responsible for and what the current run produced.</div>',
        unsafe_allow_html=True,
    )

    phase_status = execution.get("phase_status", {}) if execution else {}
    for number, name, description in PIPELINE:
        raw_status = phase_status.get(number, "READY")
        status = {
            "DONE": "COMPLETED",
            "RUNNING": "RUNNING",
            "FAILED": "FAILED",
            "SKIPPED": "SKIPPED",
            "READY": "READY",
        }.get(raw_status, raw_status)

        st.markdown(
            f"""
            <div class="stage {"stage-done" if raw_status == "DONE" else "stage-running" if raw_status == "RUNNING" else "stage-fail" if raw_status == "FAILED" else "stage-skipped" if raw_status == "SKIPPED" else ""}">
                <div style="display:flex;justify-content:space-between;gap:20px;">
                    <div>
                        <div class="stage-num">{number}</div>
                        <div class="stage-name">{esc(name)}</div>
                        <div class="stage-desc">{esc(description)}</div>
                    </div>
                    <span class="badge badge-ok">{status}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Autonomous History</div>', unsafe_allow_html=True)
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.info("No iteration history was parsed from the orchestrator output.")


# ============================================================================
# MODEL INTELLIGENCE
# ============================================================================

elif st.session_state.selected_nav == "Model Evaluation":

    st.markdown('<div class="section-title">Model Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Technical quality of the selected discovery candidate.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(
            f"""
            <div class="biz-callout">
                <div class="card-label">SELECTED PROCESS MODEL</div>
                <div class="big-value">{esc(final_model)}</div>
                <div class="small-note">Resolved from the real orchestrator execution.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="biz-callout">
                <div class="card-label">COMPOSITE SELECTION SCORE</div>
                <div class="big-value">
                    {f"{selection_score:.4f}" if selection_score is not None else "—"}
                </div>
                <div class="small-note">
                    Captured from the model-selection output when available.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metrics = [
        ("Fitness", fitness, "How well the model represents observed behavior."),
        ("Precision", precision, "How much unnecessary behavior the model permits."),
        ("Generalization", generalization, "Ability to represent unseen valid behavior."),
        ("Simplicity", simplicity, "Structural complexity / interpretability."),
    ]

    st.markdown('<div class="section-title">Quality Dimensions</div>', unsafe_allow_html=True)

    for name, value, description in metrics:
        if value is None:
            continue
        st.markdown(f"**{name}: {value:.4f}**")
        st.progress(max(0.0, min(1.0, float(value))))
        st.caption(description)


# ============================================================================
# RAG KNOWLEDGE
# ============================================================================

elif st.session_state.selected_nav == "RAG Knowledge":

    render_rag_knowledge_page()


# ============================================================================
# AI REASONING
# ============================================================================

elif st.session_state.selected_nav == "AI Reasoning":

    st.markdown(
        '<div class="section-title">AI / LLM Reasoning Layer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Functional interpretation of the autonomous decision captured from the execution.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="biz-callout">
            <div class="biz-heading">Autonomous decision</div>
            <div class="biz-body">
                The reasoning layer selected <b>{esc(action)}</b> using strategy
                <b>{esc(strategy)}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">RECOMMENDED ACTION</div>
                <div class="big-value">{esc(action)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">ADAPTATION STRATEGY</div>
                <div class="big-value">{esc(strategy)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="decision">
            <div class="decision-title">Reasoning captured from the autonomous engine</div>
            <div class="decision-text">{esc(reason)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # RAG / Groq evidence captured from the real reasoning-agent run.
    st.markdown('<div class="section-title">RAG + LLM Evidence</div>', unsafe_allow_html=True)

    rag_col1, rag_col2, rag_col3 = st.columns(3)

    with rag_col1:
        st.markdown(
            metric_card(
                "RAG STATUS",
                "CONNECTED" if rag_enabled else "NOT CONNECTED",
                "Persistent knowledge base used by the reasoning agent.",
            ),
            unsafe_allow_html=True,
        )

    with rag_col2:
        st.markdown(
            metric_card(
                "RAG CHUNKS",
                str(rag_chunks) if rag_chunks is not None else "—",
                "Knowledge chunks retrieved for this reasoning cycle.",
            ),
            unsafe_allow_html=True,
        )

    with rag_col3:
        groq_note = (
            f"{groq_response_time:.2f}s response"
            if groq_response_time is not None
            else "Response status"
        )
        st.markdown(
            metric_card(
                "GROQ",
                "SUCCESS" if groq_success else "NOT CONFIRMED",
                groq_note,
            ),
            unsafe_allow_html=True,
        )

    if rag_sources:
        st.markdown('<div class="section-title">Retrieved Knowledge</div>', unsafe_allow_html=True)
        rag_df = pd.DataFrame(rag_sources)
        rag_df = rag_df.rename(
            columns={"source": "Source", "score": "Similarity Score"}
        )
        st.dataframe(
            rag_df,
            use_container_width=True,
            hide_index=True,
        )
    elif execution is not None:
        st.info("No RAG source list was detected in the captured orchestrator output.")

    if execution is None:
        st.info(
            "No APMOA execution has been run yet. Run APMOA from the sidebar to populate the reasoning result."
        )


# ============================================================================
# AUTONOMOUS LOOP
# ============================================================================

elif st.session_state.selected_nav == "Autonomous Adaptation":

    st.markdown('<div class="section-title">Autonomous Feedback Loop</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">How APMOA moved from discovery to evaluation, reasoning and adaptation.</div>',
        unsafe_allow_html=True,
    )

    flow = [
        ("01", "DISCOVER", "Generate candidate models"),
        ("02", "EVALUATE", "Measure quality"),
        ("03", "SELECT", "Choose strongest candidate"),
        ("04", "REASON", "Interpret weaknesses"),
        ("05", "ADAPT", "Keep or rediscover"),
        ("06", "ANALYZE", "Find process issues"),
        ("07", "OPTIMIZE", "Generate actions"),
    ]

    cols = st.columns(len(flow))
    for col, (num, title, body) in zip(cols, flow):
        with col:
            st.markdown(
                f"""
                <div class="card" style="min-height:155px;">
                    <div class="card-label">{num}</div>
                    <div class="card-title" style="margin-top:8px;">{title}</div>
                    <div class="decision-text">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    action_lower = str(action).lower()
    if "keep" in action_lower:
        st.success(f"KEEP_MODEL — {final_model} was retained.")
    elif "rediscover" in action_lower:
        st.warning(f"REDISCOVER — next strategy: {strategy}.")
    else:
        st.info(f"Final autonomous action: {action}")

    if history:
        st.markdown('<div class="section-title">Iteration History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


# ============================================================================
# PROCESS INTELLIGENCE
# ============================================================================

elif st.session_state.selected_nav == "Process Intelligence":

    st.markdown('<div class="section-title">Process Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Operational findings extracted from the optimization / process-intelligence output.</div>',
        unsafe_allow_html=True,
    )

    # Explicit process-intelligence counters are parsed separately from the
    # recommendation list so the dashboard reflects the actual backend summary.
    pi_cols = st.columns(4)

    for col, label, value, note in [
        (pi_cols[0], "BOTTLENECKS", bottlenecks, "Potential bottleneck activities"),
        (pi_cols[1], "DEVIATIONS", deviations, "Potential process deviations"),
        (pi_cols[2], "REWORK", rework_activities, "Activities with rework"),
        (pi_cols[3], "TRANSITIONS", transitions_analyzed, "Transitions analyzed"),
    ]:
        with col:
            st.markdown(
                metric_card(
                    label,
                    str(value) if value is not None else "—",
                    note,
                ),
                unsafe_allow_html=True,
            )

    if not recommendations:
        st.info(
            "No structured bottleneck/deviation recommendations were detected "
            "in the captured output."
        )
    else:
        priorities = ["HIGH", "MEDIUM", "LOW"]
        c1, c2, c3 = st.columns(3)
        for col, priority in zip([c1, c2, c3], priorities):
            count = sum(r["priority"] == priority for r in recommendations)
            with col:
                st.markdown(
                    metric_card(priority, str(count), "Findings"),
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-title">Operational Findings</div>', unsafe_allow_html=True)

        for rec in recommendations:
            priority = rec["priority"]
            badge_cls = (
                "badge-fail" if priority == "HIGH"
                else "badge-warn" if priority == "MEDIUM"
                else "badge-ok"
            )
            st.markdown(
                f"""
                <div class="rec">
                    <div class="rec-id">{esc(rec["id"])}</div>
                    <div class="rec-title">{esc(rec["category"])}</div>
                    <div class="rec-meta">
                        <span class="badge {badge_cls}">{priority}</span>
                        &nbsp; Activity: {esc(rec["activity"])}
                    </div>
                    <div class="rec-body">{esc(rec["action"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# OPTIMIZATION
# ============================================================================

elif st.session_state.selected_nav == "Optimization":

    st.markdown('<div class="section-title">Optimization Command Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Filter and review the actionable recommendations produced by the optimization agent.</div>',
        unsafe_allow_html=True,
    )

    optimization_cols = st.columns(4)

    for col, label, value, note in [
        (
            optimization_cols[0],
            "TOTAL",
            optimization_recommendation_count,
            "Recommendations generated",
        ),
        (
            optimization_cols[1],
            "HIGH",
            high_priority,
            "High-priority actions",
        ),
        (
            optimization_cols[2],
            "MEDIUM",
            medium_priority,
            "Medium-priority actions",
        ),
        (
            optimization_cols[3],
            "LOW",
            low_priority,
            "Low-priority actions",
        ),
    ]:
        with col:
            st.markdown(
                metric_card(
                    label,
                    str(value) if value is not None else "—",
                    note,
                ),
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Optimization Recommendations</div>',
        unsafe_allow_html=True,
    )

    if not recommendations:
        st.info("No structured optimization recommendations were found.")
    else:
        priorities = ["ALL", "HIGH", "MEDIUM", "LOW"]
        selected = st.selectbox("Priority filter", priorities)

        filtered = (
            recommendations
            if selected == "ALL"
            else [r for r in recommendations if r["priority"] == selected]
        )

        st.caption(f"Showing {len(filtered)} of {len(recommendations)} recommendation(s).")

        for rec in filtered:
            priority = rec["priority"]
            badge_cls = (
                "badge-fail" if priority == "HIGH"
                else "badge-warn" if priority == "MEDIUM"
                else "badge-ok"
            )

            with st.expander(
                f'{rec["id"]}  |  {priority}  |  {rec["category"]}  |  {rec["activity"]}',
                expanded=(priority == "HIGH"),
            ):
                st.markdown(
                    f"""
                    <div class="rec" style="margin-bottom:0;">
                        <div class="rec-id">{esc(rec["id"])}</div>
                        <div class="rec-title">{esc(rec["category"])}</div>
                        <div class="rec-meta">
                            <span class="badge {badge_cls}">{priority}</span>
                        </div>
                        <div class="rec-body">
                            <b>Activity:</b> {esc(rec["activity"])}<br><br>
                            <b>Recommended action:</b> {esc(rec["action"])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.download_button(
            "Download Recommendations CSV",
            data=pd.DataFrame(recommendations).to_csv(index=False),
            file_name="apmoa_recommendations.csv",
            mime="text/csv",
        )


# ============================================================================
# BUSINESS VIEW
# ============================================================================

elif st.session_state.selected_nav == "Business View":

    st.markdown('<div class="section-title">Functional / Business Decision View</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">A process-owner view of what the technical output means operationally.</div>',
        unsafe_allow_html=True,
    )

    finding_count = len(recommendations)

    st.markdown(
        f"""
        <div class="biz-callout">
            <div class="biz-heading">Executive interpretation</div>
            <div class="biz-body">
                APMOA evaluated process behavior and selected
                <b>{esc(final_model)}</b> as the final model after
                <b>{iterations}</b> autonomous iteration(s).
                The model achieved fitness of
                <b>{f"{fitness:.4f}" if fitness is not None else "N/A"}</b>
                and precision of
                <b>{f"{precision:.4f}" if precision is not None else "N/A"}</b>.
                The autonomous decision was <b>{esc(action)}</b>.
                The optimization layer identified <b>{finding_count}</b>
                structured finding(s).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Operational Evidence</div>', unsafe_allow_html=True)

    evidence_cols = st.columns(4)

    for col, label, value, note in [
        (evidence_cols[0], "BOTTLENECKS", bottlenecks, "Potential bottlenecks"),
        (evidence_cols[1], "DEVIATIONS", deviations, "Potential deviations"),
        (evidence_cols[2], "REWORK", rework_activities, "Activities with rework"),
        (
            evidence_cols[3],
            "OPTIMIZATION",
            optimization_recommendation_count,
            "Generated recommendations",
        ),
    ]:
        with col:
            st.markdown(
                metric_card(
                    label,
                    str(value) if value is not None else "—",
                    note,
                ),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">What this means by function</div>', unsafe_allow_html=True)

    function_rows = [
        (
            "Process Owner",
            "Understand where actual execution is creating operational friction and which activities require attention.",
        ),
        (
            "Operations",
            "Use bottleneck and activity-level recommendations to target workload, waiting time and execution inefficiencies.",
        ),
        (
            "Process Excellence",
            "Assess whether identified deviations represent legitimate business exceptions or opportunities to redesign the process.",
        ),
        (
            "Management",
            "Prioritize improvement initiatives using measurable evidence rather than isolated anecdotal observations.",
        ),
        (
            "IT / Automation",
            "Use the selected model and optimization findings as inputs for workflow, automation and system-improvement decisions.",
        ),
        (
            "AI / Decision Intelligence",
            "Use LLM reasoning, Crew-based specialization and RAG knowledge to connect process evidence with policies and business context.",
        ),
    ]

    for title, body in function_rows:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">{esc(title)}</div>
                <div class="decision-text">{esc(body)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if recommendations:
        st.markdown('<div class="section-title">Top Actionable Opportunities</div>', unsafe_allow_html=True)
        for rec in recommendations[:5]:
            st.markdown(
                f"""
                <div class="decision">
                    <div class="decision-title">
                        {esc(rec["id"])} — {esc(rec["category"])}
                    </div>
                    <div class="decision-text">
                        <b>Activity:</b> {esc(rec["activity"])}<br>
                        <b>Action:</b> {esc(rec["action"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# EXECUTION LOG
# ============================================================================

elif st.session_state.selected_nav == "Datasets":
    render_datasets_page()

elif st.session_state.selected_nav == "Experiments":
    render_persistent_experiments_page()

elif st.session_state.selected_nav == "Baseline vs APMOA":
    render_baseline_vs_apmoa()

elif st.session_state.selected_nav == "Multi-Dataset Comparison":
    render_multi_dataset_comparison()

elif st.session_state.selected_nav == "What-If Analysis":
    render_what_if()

elif st.session_state.selected_nav == "Settings":
    render_settings()

elif st.session_state.selected_nav == "Execution Log":

    st.markdown('<div class="section-title">Execution Trace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">The exact stdout/stderr captured from the isolated APMOA backend process. The same backend lines are streamed live to the terminal while APMOA runs.</div>',
        unsafe_allow_html=True,
    )

    if execution is None:
        st.info(
            "No APMOA execution has been started yet. "
            "Click RUN APMOA from the sidebar to start the backend."
        )
    else:
        if execution.get("success", False):
            st.success(
                f"Backend completed successfully in "
                f"{execution.get('duration', 0.0):.2f}s."
            )
        else:
            st.error(
                f"Backend exited with return code "
                f"{execution.get('return_code', -1)}."
            )

        stdout = execution.get("stdout", "")
        stderr = execution.get("stderr", "")

        if stdout:
            with st.expander("Standard output", expanded=True):
                st.markdown(
                    f'<div class="log">{html.escape(stdout)}</div>',
                    unsafe_allow_html=True,
                )

        if stderr:
            with st.expander("Errors / stderr", expanded=True):
                st.code(stderr, language="text")

        st.download_button(
            "Download Complete Execution Log",
            data=execution.get("combined", ""),
            file_name="apmoa_execution_log.txt",
            mime="text/plain",
        )


# ============================================================================
# FOOTER
# ============================================================================

# Streamlit renders the dashboard before the first run, so execution can be
# None. Keep the footer safe in that state.
if execution is not None:
    duration = execution.get("duration", 0.0) or 0.0
else:
    duration = 0.0

st.markdown(
    f"""
    <div class="footer">
        APMOA • Autonomous Process Mining & Optimization
        &nbsp; | &nbsp;
        Execution time: {duration:.2f}s
        &nbsp; | &nbsp;
        {datetime.now().strftime("%d %b %Y • %H:%M:%S")}
    </div>
    """,
    unsafe_allow_html=True,
)
