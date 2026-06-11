"""
agent.py

LangGraph agent for ESM2 variant effect prediction.

Graph structure:
    fetch_protein_info → score_variants → summarize_and_report

The agent receives a gene name and list of mutations, runs ESM2 scoring,
fetches UniProt annotation, and generates a structured markdown report.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

from src.tools import score_variants, fetch_uniprot, summarize_results
from src.prompts import AGENT_SYSTEM_PROMPT

load_dotenv()
MODEL_NAME = "claude-haiku-4-5-20251001"

# --- State -------------------------------------------------------------------

class AgentState(TypedDict):
    """State passed between nodes in the LangGraph graph."""
    gene_name: str
    wildtype_sequence: str
    mutations: list[str]
    messages: Annotated[list, "append"]
    uniprot_info: dict
    scored_mutations: list[dict]
    report: str



# --- Nodes -------------------------------------------------------------------

def fetch_protein_node(state: AgentState) -> AgentState:
    """
    Node 1: Fetch UniProt annotation for the gene.
    Calls fetch_uniprot tool directly (no LLM needed for this step).
    """
    result = fetch_uniprot.invoke({"gene_name": state["gene_name"]})
    return {"uniprot_info": result}


def score_variants_node(state: AgentState) -> AgentState:
    """
    Node 2: Score mutations with ESM2.
    Calls score_variants tool directly with wildtype sequence and mutation list.
    """
    result = score_variants.invoke({
        "wildtype_sequence": state["wildtype_sequence"],
        "mutations": state["mutations"],
    })
    return {"scored_mutations": result["scored_mutations"]}


def report_node(state: AgentState) -> AgentState:
    """
    Node 3: Generate structured report.
    Passes scoring results and UniProt context to Claude for interpretation.
    Includes retry logic for API overload (529 errors).
    """
    import time
    from anthropic import OverloadedError

    context = summarize_results.invoke({
        "scored_mutations": state["scored_mutations"],
        "uniprot_info": state["uniprot_info"],
    })

    llm = ChatAnthropic(
        model=MODEL_NAME,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2000,
    )
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Generate a variant effect report based on the following ESM2 scoring results:

{context}

Gene: {state['gene_name']}
Mutations analyzed: {', '.join(state['mutations'][:10])}{'...' if len(state['mutations']) > 10 else ''}
""")
    ]

    # Retry up to 3 times with exponential backoff on 529
    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            return {"report": response.content, "messages": [response]}
        except OverloadedError:
            wait = 10 * (attempt + 1)
            print(f"API overloaded, retrying in {wait}s... (attempt {attempt + 1}/3)")
            time.sleep(wait)

    return {"report": "Error: API overloaded after 3 retries.", "messages": []}


# --- Graph -------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("fetch_protein", fetch_protein_node)
    graph.add_node("score_variants", score_variants_node)
    graph.add_node("generate_report", report_node)

    # Add edges — linear pipeline
    graph.set_entry_point("fetch_protein")
    graph.add_edge("fetch_protein", "score_variants")
    graph.add_edge("score_variants", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# --- Run ---------------------------------------------------------------------

def run_agent(gene_name: str, wildtype_sequence: str, mutations: list[str]) -> str:
    """
    Run the variant effect agent.

    Args:
        gene_name: gene symbol (e.g. 'BLAT', 'BRCA1')
        wildtype_sequence: wildtype protein sequence
        mutations: list of mutations in format 'X123Y'

    Returns:
        Structured markdown report as string
    """
    graph = build_graph()

    initial_state = {
        "gene_name": gene_name,
        "wildtype_sequence": wildtype_sequence,
        "mutations": mutations,
        "messages": [],
        "uniprot_info": {},
        "scored_mutations": [],
        "report": "",
    }

    result = graph.invoke(initial_state)
    return result["report"]


if __name__ == "__main__":
    # Quick test with BLAT_ECOLX data
    from pathlib import Path
    import pandas as pd

    df = pd.read_csv(Path("data/BLAT_ECOLX_Stiffler_2015.csv"))

    # Reconstruct wildtype
    mut = df.iloc[0]['mutant']
    wt_seq = list(df.iloc[0]['mutated_sequence'])
    wt_seq[int(mut[1:-1]) - 1] = mut[0]
    wt_sequence = ''.join(wt_seq)

    # Sample 10 mutations
    mutations = df.sample(10, random_state=42)['mutant'].tolist()

    print("Running ESM2 variant effect agent...")
    report = run_agent("BLAT", wt_sequence, mutations)
    print(report)