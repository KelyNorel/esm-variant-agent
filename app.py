"""
app.py

Streamlit UI for the ESM2 variant effect agent.
Input: gene name, wildtype sequence, mutations
Output: structured variant effect report
"""

import streamlit as st
from pathlib import Path
import pandas as pd
from src.agent import run_agent
from src.literature import MAX_MUTATIONS, fetch_clinical_evidence

# --- Page config -------------------------------------------------------------

st.set_page_config(
    page_title="ESM2 Variant Effect Agent",
    page_icon="🧬",
    layout="wide",
)
st.markdown("""
<style>
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1a6a8a;
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 ESM2 Variant Effect Agent")
st.caption("Zero-shot protein variant scoring with ESM2 + LangGraph + Claude")

# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("Search Available Proteins")
    search_term = st.text_input("Search by gene name", placeholder="e.g. BRCA, TP53, PTEN")

    if search_term:
        available = sorted([f.stem for f in Path("data").glob("*.csv")])
        matches = [f for f in available if search_term.upper() in f.upper()]
        if matches:
            for match in matches[:10]:
                gene = match.split("_")[0]
                if st.button(match, key=f"search_{match}"):
                    data_path = Path(f"data/{match}.csv")
                    df = pd.read_csv(data_path)
                    mut = df.iloc[0]['mutant']
                    wt_seq = list(df.iloc[0]['mutated_sequence'])
                    wt_seq[int(mut[1:-1]) - 1] = mut[0]
                    wt_sequence = ''.join(wt_seq)
                    sample_mutations = df.sample(10, random_state=42)['mutant'].tolist()
                    st.session_state['gene_name'] = gene
                    st.session_state['wildtype_sequence'] = wt_sequence
                    st.session_state['mutations_text'] = '\n'.join(sample_mutations)
                    st.session_state['dataset_name'] = match
                    st.rerun()
        else:
            st.warning(f"No datasets found for '{search_term}'")

    st.divider()
    st.markdown("**Quick examples:**")

    for label, gene, filename in [
        ("BLAT_ECOLX", "BLAT", "BLAT_ECOLX_Stiffler_2015.csv"),
        ("BRCA1", "BRCA1", "BRCA1_HUMAN_Findlay_2018.csv"),
        ("PTEN", "PTEN", "PTEN_HUMAN_Matreyek_2021.csv"),
    ]:
        if st.button(f"Load {label}", key=f"example_{label}"):
            data_path = Path(f"data/{filename}")
            if data_path.exists():
                df = pd.read_csv(data_path)
                mut = df.iloc[0]['mutant']
                wt_seq = list(df.iloc[0]['mutated_sequence'])
                wt_seq[int(mut[1:-1]) - 1] = mut[0]
                wt_sequence = ''.join(wt_seq)
                sample_mutations = df.sample(10, random_state=42)['mutant'].tolist()
                st.session_state['gene_name'] = gene
                st.session_state['wildtype_sequence'] = wt_sequence
                st.session_state['mutations_text'] = '\n'.join(sample_mutations)
                st.session_state['dataset_name'] = filename.replace('.csv', '')
                st.rerun()
            else:
                st.error(f"{filename} not found.")

    st.divider()
    st.markdown("**Model:** `facebook/esm2_t6_8M_UR50D`")
    st.markdown("**Benchmark:** Spearman ρ = 0.364 on ProteinGym")
    st.markdown("**Device:** MPS (Apple Silicon)")

# --- Main form ---------------------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:
    gene_name = st.text_input(
        "Gene name",
        value=st.session_state.get('gene_name', ''),
        placeholder="e.g. BRCA1, TP53, BLAT",
    )
    wildtype_sequence = st.text_area(
        "Wildtype protein sequence",
        value=st.session_state.get('wildtype_sequence', ''),
        placeholder="MSIQHFRVALIPFF...",
        height=150,
    )

with col2:
    mutations_text = st.text_area(
        "Mutations (one per line, format: X123Y)",
        value=st.session_state.get('mutations_text', ''),
        placeholder="A24G\nR56W\nF70E",
        height=150,
    )
    st.markdown("**Format:** `X123Y` = wildtype AA + position + mutant AA")
    st.markdown("Example: `F70E` = Phe→Glu at position 70")

# --- Run agent ---------------------------------------------------------------

run_button = st.button("Run Variant Effect Analysis", type="primary", use_container_width=True)

if run_button:
    if not gene_name:
        st.error("Please enter a gene name.")
        st.stop()
    if not wildtype_sequence:
        st.error("Please enter a wildtype sequence.")
        st.stop()
    if not mutations_text:
        st.error("Please enter at least one mutation.")
        st.stop()

    mutations = [m.strip() for m in mutations_text.strip().split('\n') if m.strip()]

    if len(mutations) == 0:
        st.error("No valid mutations found.")
        st.stop()

    with st.spinner("Running ESM2 variant effect analysis..."):
        try:
            result = run_agent(gene_name, wildtype_sequence, mutations,
                               dataset_name=st.session_state.get('dataset_name', ''))
            st.session_state['report'] = result["report"]
            st.session_state['scored_mutations'] = result["scored_mutations"]
        except Exception as e:
            st.error(f"Agent error: {str(e)}")
            st.stop()

# --- Display report — always from session_state ------------------------------

report = st.session_state.get('report', '')

if report:
    st.divider()
    st.markdown(report)
    st.download_button(
        label="Download Report",
        data=report,
        file_name=f"variant_report_{gene_name}.md",
        mime="text/markdown",
    )

    # --- Clinical Evidence ---------------------------------------------------

    st.divider()
    st.subheader("🔬 Clinical Evidence Search")
    st.caption("Search ClinVar, PubMed, and UniProt for selected mutations.")

    highly_del = [
        m['mutant'] for m in st.session_state.get('scored_mutations', [])
        if m['llr'] < -2.0
    ][:MAX_MUTATIONS]

    st.markdown(f"Highly deleterious mutations pre-selected (LLR < −2.0). **Max {MAX_MUTATIONS}.**")

    evidence_mutations = st.text_area(
        "Mutations to investigate (one per line, max 5):",
        value='\n'.join(highly_del),
        height=120,
        key="evidence_mutations"
    )

    if st.button("Fetch Clinical Evidence", key="fetch_evidence"):
        mutations_to_check = [m.strip() for m in evidence_mutations.strip().split('\n') if m.strip()]

        if len(mutations_to_check) == 0:
            st.warning("No mutations entered.")
        elif len(mutations_to_check) > MAX_MUTATIONS:
            st.warning(f"Maximum {MAX_MUTATIONS} mutations allowed. Using first {MAX_MUTATIONS}.")
            mutations_to_check = mutations_to_check[:MAX_MUTATIONS]

        with st.spinner(f"Searching literature for {len(mutations_to_check)} mutations..."):
            evidence = fetch_clinical_evidence(gene_name, mutations_to_check)

        st.session_state['evidence'] = evidence

    # Display evidence — always from session_state
    evidence = st.session_state.get('evidence', '')
    if evidence:
        st.markdown(evidence)
        st.download_button(
            label="Download Clinical Evidence",
            data=evidence,
            file_name=f"clinical_evidence_{gene_name}.md",
            mime="text/markdown",
            key="download_evidence"
        )
