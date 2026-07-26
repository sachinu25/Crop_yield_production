import streamlit as st
import pandas as pd

def render():
    """Render the Prediction History page."""
    st.markdown('<div class="section-header">📜 Prediction History</div>', unsafe_allow_html=True)
    
    if "history" not in st.session_state or not st.session_state.history:
        html = """
        <div class="empty-state">
            <span class="empty-icon">📂</span>
            <div class="empty-text">No predictions made in this session yet.</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">Head over to the Prediction page to get started.</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        return
        
    df = pd.DataFrame(st.session_state.history)
    
    st.markdown("Your predictions from the current session are saved here.")
    
    # Render premium dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Download CSV",
            data=csv,
            file_name="crop_predictions_history.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with c2:
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
