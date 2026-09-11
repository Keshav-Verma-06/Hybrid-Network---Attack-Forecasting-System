"""Shared presentation helpers for the NetSentinel Streamlit workspace."""

import streamlit as st


COLORS = {
    "ink": "#191c1e",
    "muted": "#5f6368",
    "line": "#c5c6cd",
    "surface": "#ffffff",
    "canvas": "#f8f9fb",
    "soft": "#edeef0",
    "primary": "#37455c",
    "success": "#3f7d58",
    "warning": "#a87524",
    "danger": "#b54747",
}


def apply_theme() -> None:
    """Apply the Stitch-inspired shell without changing Streamlit behavior."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        :root { --ink:#191c1e; --muted:#5f6368; --line:#c5c6cd; --surface:#fff; --canvas:#f8f9fb; --soft:#edeef0; --primary:#37455c; --success:#3f7d58; --warning:#a87524; --danger:#b54747; }
        html, body, [class*="css"] { font-family: 'Geist', sans-serif; }
        [data-testid="stAppViewContainer"] { background: var(--canvas); }
        [data-testid="stHeader"] { background: rgba(255,255,255,.94); border-bottom: 1px solid rgba(197,198,205,.55); }
        [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid rgba(197,198,205,.55); }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
        [data-testid="stSidebarNav"] { padding-top: .5rem; }
        [data-testid="stSidebarNav"] span { font-family: 'Geist', sans-serif; }
        [data-testid="stMetric"] { background: var(--surface); border: 1px solid rgba(197,198,205,.55); border-radius: 8px; padding: 1rem; }
        [data-testid="stMetricLabel"] { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }
        [data-testid="stMetricValue"] { color: var(--ink); font-family: 'JetBrains Mono', monospace; }
        .ns-card { background: var(--surface); border: 1px solid rgba(197,198,205,.55); border-radius: 10px; padding: 1.25rem; box-shadow: 0 2px 8px rgba(25,28,30,.03); }
        .ns-kicker { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; }
        .ns-title { color: var(--ink); font-size: 2rem; font-weight: 600; letter-spacing: -.03em; margin: .25rem 0 .25rem; }
        .ns-subtitle { color: var(--muted); font-size: .95rem; margin-bottom: 1.5rem; }
        .ns-label { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }
        .ns-risk { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; font-weight: 600; line-height: 1; color: var(--ink); }
        .ns-risk span { color: var(--muted); font-size: 1.25rem; font-weight: 400; }
        .ns-badge { display:inline-flex; align-items:center; border-radius:4px; border:1px solid; padding:.2rem .5rem; font-family:'JetBrains Mono',monospace; font-size:.67rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
        .ns-badge.good { color: var(--success); background:#eef7f0; border-color:#b8d9c0; }
        .ns-badge.warn { color: var(--warning); background:#faf3e7; border-color:#e0c393; }
        .ns-badge.bad { color: var(--danger); background:#faeded; border-color:#e39f9f; }
        .ns-status { display:flex; align-items:center; gap:.5rem; color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; }
        .ns-dot { width:8px; height:8px; border-radius:999px; display:inline-block; }
        .ns-dot.good { background:var(--success); } .ns-dot.warn { background:var(--warning); } .ns-dot.bad { background:var(--danger); }
        div[data-testid="stPageLink"] a { border-radius: 6px; }
        div[data-testid="stPageLink"] a:hover { background: var(--soft); }
        button[kind="primary"] { background: var(--primary); border-color: var(--primary); }
        footer { visibility: hidden; }
        @media (max-width: 720px) { .ns-title { font-size: 1.65rem; } .ns-risk { font-size: 2.8rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_start() -> None:
    st.markdown('<section class="ns-card">', unsafe_allow_html=True)


def card_end() -> None:
    st.markdown('</section>', unsafe_allow_html=True)


def badge(label: str, tone: str = "good") -> str:
    return f'<span class="ns-badge {tone}">{label}</span>'
