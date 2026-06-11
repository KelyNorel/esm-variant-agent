"""
esm_runner.py

Loads ESM2 and computes log-likelihood ratio (LLR) scores for protein variants.
Core scoring module used by the LangGraph agent.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# Default model - small and fast, practical for agentic inference
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
MAX_SEQ_LENGTH = 1022  # ESM2 hard limit (1024 - 2 special tokens)


# ESMRunner is implemented as a class (rather than functions) because ESM2 takes
# ~2 seconds to load. The class loads the model once in __init__ and reuses it
# across all scoring calls in an agent session — important for interactive use.
class ESMRunner:
    """
    Wrapper around ESM2 for zero-shot variant effect prediction.
    
    Uses masked language modeling to compute log-likelihood ratios (LLR)
    for single amino acid substitutions. A negative LLR indicates the mutant
    amino acid is less probable than wildtype given sequence context.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model identifier for ESM2
            device: 'mps', 'cuda', or 'cpu'. Auto-detected if None.
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.model_name = model_name

        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model ready — {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def compute_llr(self, sequence: str, position: int, mutant_aa: str, wildtype_aa: str) -> float:
        """
        Compute LLR for a single amino acid substitution.

        Args:
            sequence: wildtype protein sequence (str)
            position: 1-indexed mutation position
            mutant_aa: mutant amino acid (single letter code)
            wildtype_aa: wildtype amino acid (single letter code)

        Returns:
            LLR = log P(mutant | context) - log P(wildtype | context)
            Negative values indicate predicted deleterious effect.
        """
        if len(sequence) > MAX_SEQ_LENGTH:
            print(f"Warning: sequence length {len(sequence)} exceeds {MAX_SEQ_LENGTH}. Truncating.")
            sequence = sequence[:MAX_SEQ_LENGTH]

        # Mask target position
        masked = list(sequence)
        masked[position - 1] = self.tokenizer.mask_token
        masked_seq = ''.join(masked)

        # Tokenize and move to device
        inputs = self.tokenizer(masked_seq, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract log-probabilities at masked position
        logits = outputs.logits[0, position, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        wt_id = self.tokenizer.convert_tokens_to_ids(wildtype_aa)
        mut_id = self.tokenizer.convert_tokens_to_ids(mutant_aa)

        return (log_probs[mut_id] - log_probs[wt_id]).item()

    def score_mutations(self, wildtype_sequence: str, mutations: list[str]) -> list[dict]:
        """
        Score a list of mutations against a wildtype sequence.

        Args:
            wildtype_sequence: wildtype protein sequence
            mutations: list of mutation strings in format 'X123Y'
                       (wildtype_aa + position + mutant_aa)

        Returns:
            List of dicts with keys: mutant, position, wt_aa, mut_aa, llr, predicted_effect
        """
        results = []
        for mut in mutations:
            wt_aa = mut[0]
            position = int(mut[1:-1])
            mut_aa = mut[-1]

            llr = self.compute_llr(wildtype_sequence, position, mut_aa, wt_aa)
            results.append({
                'mutant': mut,
                'position': position,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'llr': round(llr, 4),
                'predicted_effect': 'deleterious' if llr < 0 else 'neutral/beneficial'
            })

        return results