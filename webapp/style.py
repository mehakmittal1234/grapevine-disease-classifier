"""Design tokens, global CSS and small HTML helpers shared by every page.

Palette: limestone vineyard soil, vine-leaf green, grape-must purple for the brand and the
blue-green of Bordeaux mixture (the copper fungicide sprayed on vines) for interactive accents.
"""

from __future__ import annotations

import streamlit as st

INK = "#16201B"
MUTED = "#56625A"
LINE = "#D5DBD1"
LIMESTONE = "#EEF0EA"
PAPER = "#FFFFFF"
GRAPE = "#5B2A57"
COPPER = "#1D7F8E"
LEAF = "#3E6B35"

DISPLAY_FONT = "Bricolage Grotesque"
BODY_FONT = "Instrument Sans"

CLASSES = {
    "Black Rot": ("#3F2A22", "Round tan-to-brown spots with dark borders and tiny black fruiting bodies. "
                             "Fungal (Guignardia bidwellii)."),
    "ESCA": ("#B04A2B", "Tiger-stripe yellow-to-red bands between the veins that dry to brown. "
                        "A trunk disease, also called black measles."),
    "Leaf Blight": ("#9A7A0E", "Irregular dark-brown spots that merge and dry the leaf out. "
                               "Isariopsis leaf spot (Pseudocercospora vitis)."),
    "Healthy": (LEAF, "Evenly green leaf with no disease symptoms."),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap');
:root {{
  --ink: {INK}; --muted: {MUTED}; --line: {LINE}; --limestone: {LIMESTONE}; --paper: {PAPER};
  --grape: {GRAPE}; --copper: {COPPER}; --leaf: {LEAF};
  --display: '{DISPLAY_FONT}', 'Instrument Sans', system-ui, sans-serif;
  --body: '{BODY_FONT}', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
html, body, .stApp, .stMarkdown, .stCaption, button, input, label {{font-family: var(--body);}}
.stApp {{background: var(--limestone); color: var(--ink);}}
.block-container {{max-width: 1180px; padding-top: 4.2rem; padding-bottom: 3rem;}}
header[data-testid="stHeader"] {{background: rgba(238,240,234,.92); backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line);}}
h1, h2, h3, .display {{font-family: var(--display) !important; color: var(--ink); letter-spacing: -.02em;}}
a {{color: var(--copper);}}
:focus-visible {{outline: 2px solid var(--copper) !important; outline-offset: 2px;}}

/* Page intro */
h1.page-title {{padding: 0 !important; font-family: var(--display); font-weight: 800; font-size: clamp(2.1rem, 4.6vw, 3.4rem) !important;
  line-height: 1.02 !important; margin: .4rem 0 .9rem; max-width: 15ch; font-variation-settings: 'opsz' 96;}}
.page-lede {{font-size: 1.12rem; line-height: 1.55; color: var(--muted); max-width: 60ch; margin: 0 0 1.6rem;}}
h2.section-title {{font-family: var(--display); font-weight: 700; font-size: 1.45rem !important; line-height: 1.2 !important;
  margin: 2.4rem 0 .35rem; padding: 0 !important;}}
.section-lede {{color: var(--muted); max-width: 64ch; margin: 0 0 1rem; line-height: 1.55;}}

/* Field guide (diagnose page, before upload) */
.guide {{border-left: 3px solid var(--grape); padding: .2rem 0 .2rem 1.3rem;}}
.guide h3 {{font-size: 1.05rem; margin: 0 0 .6rem; padding: 0;}}
.guide-item {{display: grid; grid-template-columns: 14px 1fr; gap: 12px; padding: .7rem 0;
  border-bottom: 1px solid var(--line);}}
.guide-item:last-child {{border-bottom: 0;}}
.guide-swatch {{width: 14px; height: 14px; border-radius: 4px; margin-top: 4px;}}
.guide-item b {{display: block; font-weight: 600;}}
.guide-item span {{color: var(--muted); font-size: .93rem; line-height: 1.45;}}
.facts {{display: flex; flex-wrap: wrap; gap: .5rem 1.6rem; margin-top: 1.1rem; color: var(--muted); font-size: .92rem;}}
.facts b {{color: var(--ink); font-weight: 600;}}
.footnote {{color: var(--muted); font-size: .8rem; margin-top: .9rem; line-height: 1.5; max-width: 64ch;}}

/* Grad-CAM compare viewer */
.cmp {{--split: 0%; --alpha: .55; position: relative; width: 100%; aspect-ratio: 1 / 1; border-radius: 16px;
  overflow: hidden; background: #000; box-shadow: 0 18px 40px -24px rgba(22,32,27,.6); touch-action: none;}}
.cmp img {{position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; user-select: none; pointer-events: none;}}
.cmp .cmp-heat {{opacity: var(--alpha); clip-path: inset(0 calc(100% - var(--split)) 0 0);}}
.cmp .cmp-handle {{position: absolute; top: 0; bottom: 0; left: var(--split); width: 2px; margin-left: -1px;
  background: #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.25); pointer-events: none;}}
.cmp .cmp-handle::after {{content: ''; position: absolute; top: 50%; left: 50%; width: 34px; height: 34px;
  transform: translate(-50%,-50%); border-radius: 50%; background: #fff; box-shadow: 0 2px 10px rgba(0,0,0,.35);
  background-image: linear-gradient(90deg, transparent 13px, {INK} 13px, {INK} 15px, transparent 15px, transparent 19px, {INK} 19px, {INK} 21px, transparent 21px);}}
.cmp .cmp-tag {{position: absolute; bottom: 12px; padding: 4px 10px; border-radius: 999px; font-size: .78rem;
  background: rgba(22,32,27,.72); color: #fff; pointer-events: none;}}
.cmp .cmp-tag.l {{left: 12px;}} .cmp .cmp-tag.r {{right: 12px;}}
.cmp input.cmp-split {{position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: ew-resize;}}
.cmp-tools {{display: flex; align-items: center; gap: 12px; margin-top: .8rem; color: var(--muted); font-size: .88rem;}}
.cmp-tools input {{flex: 1; accent-color: var(--copper);}}
.cmp-scan {{position: absolute; top: 0; bottom: 0; width: 80px; left: -80px; pointer-events: none;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.35), transparent); animation: scan 1.3s ease-out .1s 1 forwards;}}
@keyframes scan {{to {{left: 110%;}}}}

/* Diagnosis */
.dx {{background: var(--paper); border-radius: 16px; padding: 1.4rem 1.5rem 1.2rem; border: 1px solid var(--line);}}
.dx-label {{color: var(--muted); font-size: .9rem; margin: 0;}}
.dx-name {{font-family: var(--display); font-weight: 800; font-size: clamp(2.2rem, 4vw, 3rem); line-height: 1;
  margin: .3rem 0 .2rem; display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap;}}
.dx-dot {{width: .5em; height: .5em; border-radius: 50%; display: inline-block; transform: translateY(-.08em);}}
.dx-conf {{font-variant-numeric: tabular-nums; font-size: 1.05rem; color: var(--ink); margin: .2rem 0 .8rem;}}
.dx-conf b {{font-family: var(--display); font-size: 1.6rem;}}
.dx-note {{color: var(--muted); line-height: 1.5; margin: 0 0 1rem;}}
.prob {{display: grid; grid-template-columns: 92px 1fr 52px; gap: 10px; align-items: center; margin: 9px 0; font-size: .92rem;}}
.prob-track {{height: 8px; background: #E6EAE2; border-radius: 999px; overflow: hidden;}}
.prob-fill {{height: 100%; border-radius: 999px; transform-origin: left; animation: grow .9s cubic-bezier(.2,.8,.2,1) both;}}
.prob-val {{text-align: right; font-variant-numeric: tabular-nums;}}
@keyframes grow {{from {{transform: scaleX(0);}}}}

/* Pipeline steps (data page) */
.steps {{list-style: none; counter-reset: s; padding: 0; margin: .4rem 0 0;}}
.steps li {{counter-increment: s; display: grid; grid-template-columns: 52px 1fr minmax(120px, auto); gap: 4px 18px;
  padding: 1.05rem 0; border-top: 1px solid var(--line); align-items: baseline;}}
.steps li::before {{content: counter(s); font-family: var(--display); font-weight: 700; font-size: 1.3rem; color: var(--grape);}}
.steps h4 {{margin: 0; padding: 0; font-family: var(--body) !important; font-weight: 600; font-size: 1.02rem; letter-spacing: 0;}}
.steps p {{grid-column: 2; margin: .25rem 0 0; color: var(--muted); line-height: 1.5; font-size: .95rem;}}
.steps .num {{grid-column: 3; grid-row: 1 / span 2; text-align: right; font-family: var(--display); font-weight: 700;
  font-size: 1.7rem; font-variant-numeric: tabular-nums; color: var(--ink);}}
.steps .num small {{display: block; font-family: var(--body); font-weight: 400; font-size: .8rem; color: var(--muted);}}
.checks {{display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px 24px; margin-top: .6rem;}}
.check {{display: flex; gap: 10px; font-size: .94rem; align-items: baseline;}}
.check i {{font-style: normal; color: var(--leaf); font-weight: 700;}}

/* Figures (models and reliability pages) */
.figure {{border-top: 3px solid var(--ink); padding-top: .7rem;}}
.figure .v {{font-family: var(--display); font-weight: 800; font-size: 2.3rem; line-height: 1.05; font-variant-numeric: tabular-nums;}}
.figure .k {{color: var(--muted); font-size: .92rem; line-height: 1.45; margin-top: .3rem;}}
.pick {{border-left: 3px solid var(--grape); padding-left: 1.1rem; margin: .5rem 0 1rem; max-width: 70ch; line-height: 1.55;}}

.site-footer {{color: var(--muted); font-size: .82rem; margin-top: 3.2rem; padding-top: 1rem; border-top: 1px solid var(--line);
  display: flex; flex-wrap: wrap; gap: .5rem 2rem; justify-content: space-between;}}

@media (max-width: 640px) {{
  .block-container {{padding-top: 3.6rem;}}
  .steps li {{grid-template-columns: 34px 1fr;}}
  .steps .num {{grid-column: 2; grid-row: auto; text-align: left; font-size: 1.3rem;}}
}}
@media (prefers-reduced-motion: reduce) {{
  .prob-fill, .cmp-scan {{animation: none;}}
}}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def intro(title: str, lede: str) -> None:
    html(f'<h1 class="page-title">{title}</h1><p class="page-lede">{lede}</p>')


def section(title: str, lede: str = "") -> None:
    html(f'<h2 class="section-title">{title}</h2>' + (f'<p class="section-lede">{lede}</p>' if lede else ""))


def figures(items: list[tuple[str, str]]) -> None:
    """A row of large numbers with explanations; columns stack on narrow screens."""
    for col, (value, key) in zip(st.columns(len(items), gap="large"), items):
        with col:
            html(f'<div class="figure"><div class="v">{value}</div><div class="k">{key}</div></div>')


def footer(repo_url: str) -> None:
    html(f'<div class="site-footer"><span>Decision support only. Confirm a diagnosis with an agronomist or '
         f'plant pathologist.</span><a href="{repo_url}">Source code and full reports on GitHub</a></div>')
