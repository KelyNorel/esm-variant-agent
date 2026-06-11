"""
literature.py

Fetches clinical and literature evidence for highly deleterious protein variants
using Claude with web search tool (Anthropic API).

Limited to 5 mutations per query to keep latency reasonable.
"""

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

MAX_MUTATIONS = 5

LITERATURE_SYSTEM_PROMPT = """You are a clinical genomics assistant specializing in protein variant interpretation.

For each mutation provided, search for clinical and functional evidence using web search.
Focus on:
- ClinVar classifications (pathogenic/benign/VUS)
- Published functional studies
- Known disease associations
- Population frequency if available (gnomAD)

For each mutation, provide a concise summary (3-5 sentences max).
If no evidence is found, say so explicitly — do not speculate.

Format your response as markdown with one section per mutation:

### {GENE} {MUTATION}
**ClinVar:** ...
**Disease association:** ...
**Functional evidence:** ...
**Population frequency:** ...
**Summary:** ...
"""


def fetch_clinical_evidence(gene_name: str, mutations: list[str]) -> str:
    """
    Fetch clinical evidence for a list of mutations using Claude + web search.

    Args:
        gene_name: gene symbol (e.g. 'BRCA1', 'PTEN')
        mutations: list of mutations in format 'X123Y' (max 5)

    Returns:
        Markdown string with clinical evidence per mutation
    """
    if len(mutations) > MAX_MUTATIONS:
        mutations = mutations[:MAX_MUTATIONS]

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    mutation_list = "\n".join([f"- {gene_name} {mut}" for mut in mutations])

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=LITERATURE_SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[
            {
                "role": "user",
                "content": f"""Search for clinical and literature evidence for these protein variants:

{mutation_list}

For each mutation, search ClinVar, PubMed, and UniProt.
Report what is known — if nothing is found, say so explicitly.
"""
            }
        ]
    )

    # Extract text blocks from response
    text_blocks = [
        block.text for block in response.content
        if hasattr(block, 'text')
    ]

    if not text_blocks:
        return "No clinical evidence found for the requested mutations."

    return "\n\n".join(text_blocks)