"""Streamlit app: diagnose a grape-leaf photo, and browse how the model was built and evaluated.

Run from the project folder:

    streamlit run app.py

Pages live in ``webapp/``; every number they show is read from the saved reports in ``artifacts/``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from webapp import data_page, diagnose, models_page, reliability_page, reports, style

st.set_page_config(page_title="Grapevine Leaf Disease Classifier", page_icon="🍇", layout="wide",
                   initial_sidebar_state="collapsed")
style.apply()
st.logo(str(Path(__file__).parent / "webapp" / "logo.svg"), size="large", link=reports.REPO_URL)

page = st.navigation(
    [
        st.Page(diagnose.render, title="Diagnose", default=True),
        st.Page(data_page.render, title="Data cleaning", url_path="data"),
        st.Page(models_page.render, title="Model comparison", url_path="models"),
        st.Page(reliability_page.render, title="Reliability", url_path="reliability"),
    ],
    position="top",
)
page.run()
style.footer(reports.REPO_URL)
