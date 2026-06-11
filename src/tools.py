"""
tools.py

LangGraph-compatible tools for the ESM2 variant effect agent.
Each tool is a function decorated with @tool that the agent can call.

Tools:
    - score_variants: runs ESM2 LLR scoring on a list of mutations
    - fetch_uniprot: retrieves functional annotation from UniProt REST API
    - summarize_results: formats scoring results for report generation
"""

import requests
from langchain_core.tools import tool
from src.esm_runner import ESMRunner
from src.scoring import summarize_scores, rank_variants


# Module-level ESMRunner instance — loaded once, reused across tool calls
_runner: ESMRunner | None = None


def get_runner() -> ESMRunner:
    """Lazy-load ESMRunner singleton."""
    global _runner
    if _runner is None:
        _runner = ESMRunner()
    return _runner


@tool
def score_variants(wildtype_sequence: str, mutations: list[str]) -> dict:
    """
    Score a list of protein variants using ESM2 log-likelihood ratios.

    Args:
        wildtype_sequence: wildtype protein sequence (single letter amino acid codes)
        mutations: list of mutations in format 'X123Y' (e.g. ['A24G', 'R56W'])

    Returns:
        dict with scored mutations, summary statistics, and top deleterious variants
    """
    runner = get_runner()
    scored = runner.score_mutations(wildtype_sequence, mutations)
    summary = summarize_scores(scored)
    top = rank_variants(scored, top_n=5).to_dict(orient='records')

    return {
        'scored_mutations': scored,
        'summary': summary,
        'top_deleterious': top,
    }


@tool
def fetch_uniprot(gene_name: str, organism: str = "human") -> dict:
    """
    Fetch functional annotation for a protein from the UniProt REST API.

    Args:
        gene_name: gene symbol (e.g. 'BRCA1', 'TP53')
        organism: organism name for filtering (default: 'human')

    Returns:
        dict with protein name, function, UniProt accession, and sequence length
    """
    # Query UniProt search API
    query = f"gene:{gene_name} AND organism_name:{organism} AND reviewed:true"
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession,protein_name,gene_names,organism_name,length,cc_function",
        "format": "json",
        "size": 1,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            return {"error": f"No UniProt entry found for gene {gene_name} in {organism}"}

        entry = data["results"][0]
        accession = entry.get("primaryAccession", "N/A")
        protein_name = (
            entry.get("proteinDescription", {})
                 .get("recommendedName", {})
                 .get("fullName", {})
                 .get("value", "N/A")
        )
        function_text = ""
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    function_text = texts[0].get("value", "")
                    break

        return {
            "accession": accession,
            "protein_name": protein_name,
            "gene_name": gene_name,
            "organism": organism,
            "sequence_length": entry.get("sequence", {}).get("length", "N/A"),
            "function": function_text[:500] if function_text else "No functional annotation available",
        }

    except requests.RequestException as e:
        return {"error": f"UniProt API request failed: {str(e)}"}


@tool
def summarize_results(scored_mutations: list[dict], uniprot_info: dict) -> str:
    """
    Format ESM2 scoring results and UniProt annotation into a context string
    for the report generator.

    Args:
        scored_mutations: output from score_variants tool
        uniprot_info: output from fetch_uniprot tool

    Returns:
        Formatted string summarizing results for Claude to interpret
    """
    summary = summarize_scores(scored_mutations)
    top = rank_variants(scored_mutations, top_n=5).to_dict(orient='records')

    lines = [
        f"Protein: {uniprot_info.get('protein_name', 'N/A')}",
        f"Gene: {uniprot_info.get('gene_name', 'N/A')}",
        f"UniProt: {uniprot_info.get('accession', 'N/A')}",
        f"Function: {uniprot_info.get('function', 'N/A')}",
        "",
        f"Variants scored: {summary['n_total']}",
        f"Predicted deleterious: {summary['n_deleterious']} ({summary['pct_deleterious']}%)",
        f"Predicted neutral: {summary['n_neutral']}",
        f"LLR mean: {summary['llr_mean']} ± {summary['llr_std']}",
        "",
        "Top 5 most deleterious variants:",
    ]
    for v in top:
        lines.append(f"  {v['mutant']}  LLR={v['llr']:.3f}")

    return "\n".join(lines)