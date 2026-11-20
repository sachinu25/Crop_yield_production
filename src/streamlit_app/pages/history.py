"""
History Page — Searchable prediction history with filters, sort, and export.

All data from session state. No fake records.
"""

import streamlit as st
import pandas as pd

from src.streamlit_app.components.cards import section_header
from src.streamlit_app.utils.pdf_report import generate_history_pdf


def render():
    """Render the Prediction History page."""
    section_header("📜 Prediction History", "All predictions from the current session")

    if "history" not in st.session_state or not st.session_state.history:
        html = """
        <div class="empty-state">
            <span class="empty-icon">📂</span>
            <div class="empty-text">No predictions made in this session yet.</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem; color: var(--text-muted);">
                Navigate to 🔮 Prediction to start forecasting crop yields.
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        return

    df = pd.DataFrame(st.session_state.history)

    # ── Filters Row ──
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            search = st.text_input("🔍 Search", placeholder="Search any field...", key="hist_search")
        with f2:
            states = ["All"] + sorted(df['State'].unique().tolist())
            sel_state = st.selectbox("State", states, key="hist_state")
        with f3:
            crops = ["All"] + sorted(df['Crop'].unique().tolist())
            sel_crop = st.selectbox("Crop", crops, key="hist_crop")
        with f4:
            seasons = ["All"] + sorted(df['Season'].unique().tolist())
            sel_season = st.selectbox("Season", seasons, key="hist_season")

    # Apply filters
    filtered = df.copy()
    if search:
        mask = filtered.astype(str).apply(lambda row: row.str.contains(search, case=False).any(), axis=1)
        filtered = filtered[mask]
    if sel_state != "All":
        filtered = filtered[filtered['State'] == sel_state]
    if sel_crop != "All":
        filtered = filtered[filtered['Crop'] == sel_crop]
    if sel_season != "All":
        filtered = filtered[filtered['Season'] == sel_season]

    # Stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Predictions", len(filtered))
    with c2:
        avg_yield = filtered['Yield (T/Ha)'].mean() if len(filtered) > 0 else 0
        st.metric("Avg Yield", f"{avg_yield:.3f} T/Ha")
    with c3:
        unique_crops = filtered['Crop'].nunique() if len(filtered) > 0 else 0
        st.metric("Unique Crops", unique_crops)

    # Table
    if filtered.empty:
        st.info("No records match the current filters.")
    else:
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Yield (T/Ha)": st.column_config.NumberColumn(format="%.3f"),
                "Production (T)": st.column_config.NumberColumn(format="%.3f"),
                "Area": st.column_config.NumberColumn(format="%.2f"),
            }
        )

    # Action Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b_pdf, b2, b3 = st.columns([1, 1, 1, 1])

    with b1:
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Download CSV",
            data=csv,
            file_name="agri_ai_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with b_pdf:
        st.download_button(
            label="📄 Download PDF",
            data=generate_history_pdf(filtered.to_dict("records")),
            file_name="agri_ai_prediction_history.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with b2:
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    with b3:
        st.metric("Records Shown", f"{len(filtered)} / {len(df)}")

    # Detail View
    if not filtered.empty:
        with st.expander("📋 View Detailed Prediction"):
            idx = st.selectbox(
                "Select prediction by timestamp",
                range(len(filtered)),
                format_func=lambda i: f"{filtered.iloc[i]['Timestamp']} — {filtered.iloc[i]['Crop']} ({filtered.iloc[i]['District']})"
            )
            selected = filtered.iloc[idx]

            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.markdown("**Input Parameters**")
                for key in ['State', 'District', 'Crop', 'Season', 'Year', 'Area', 'Unit']:
                    if key in selected:
                        st.markdown(f"- **{key}:** {selected[key]}")
            with detail_cols[1]:
                st.markdown("**Results**")
                for key in ['Yield (T/Ha)', 'Production (T)', 'Confidence', 'ID', 'Timestamp']:
                    if key in selected:
                        st.markdown(f"- **{key}:** {selected[key]}")
