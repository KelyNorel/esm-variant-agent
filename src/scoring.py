"""
scoring.py

Evaluation metrics and result ranking for variant effect predictions.
Takes LLR scores from ESMRunner and computes performance metrics
against experimental ground truth (ProteinGym DMS scores).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from typing import Optional


def evaluate_llr(
    llr_scores: list[float],
    dms_scores: list[float],
    dms_bins: Optional[list[int]] = None
) -> dict:
    """
    Evaluate LLR predictions against experimental DMS scores.

    Args:
        llr_scores: predicted LLR values from ESMRunner
        dms_scores: experimental fitness scores (continuous)
        dms_bins: optional binary labels (0=deleterious, 1=neutral)

    Returns:
        dict with Spearman rho, p-value, and optional classification metrics
    """
    llr = np.array(llr_scores)
    dms = np.array(dms_scores)

    rho, pval = spearmanr(llr, dms)
    pearson_r, pearson_p = pearsonr(llr, dms)

    results = {
        'spearman_rho': round(rho, 4),
        'spearman_pval': round(pval, 6),
        'pearson_r': round(pearson_r, 4),
        'pearson_pval': round(pearson_p, 6),
        'n_variants': len(llr),
    }

    # Classification metrics if binary labels provided
    if dms_bins is not None:
        bins = np.array(dms_bins)
        # Use LLR threshold of 0: negative = deleterious, positive = neutral
        predicted_bins = (llr >= 0).astype(int)
        accuracy = (predicted_bins == bins).mean()
        
        # Precision, recall for deleterious class (bin=0)
        tp = ((predicted_bins == 0) & (bins == 0)).sum()
        fp = ((predicted_bins == 0) & (bins == 1)).sum()
        fn = ((predicted_bins == 1) & (bins == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.update({
            'accuracy': round(accuracy, 4),
            'precision_deleterious': round(precision, 4),
            'recall_deleterious': round(recall, 4),
            'f1_deleterious': round(f1, 4),
        })

    return results


def rank_variants(scored_mutations: list[dict], top_n: int = 10) -> pd.DataFrame:
    """
    Rank mutations by predicted deleteriousness (most negative LLR first).

    Args:
        scored_mutations: output from ESMRunner.score_mutations()
        top_n: number of top variants to return

    Returns:
        DataFrame sorted by LLR ascending (most deleterious first)
    """
    df = pd.DataFrame(scored_mutations)
    df = df.sort_values('llr', ascending=True).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df.head(top_n)


def summarize_scores(scored_mutations: list[dict]) -> dict:
    """
    Generate summary statistics for a set of scored mutations.
    Used by the LangGraph agent to build the report context.

    Args:
        scored_mutations: output from ESMRunner.score_mutations()

    Returns:
        dict with summary stats for the report generator
    """
    llr_values = [m['llr'] for m in scored_mutations]
    llr_array = np.array(llr_values)

    n_deleterious = sum(1 for m in scored_mutations if m['predicted_effect'] == 'deleterious')
    n_neutral = len(scored_mutations) - n_deleterious

    top_deleterious = sorted(scored_mutations, key=lambda x: x['llr'])[:5]
    top_neutral = sorted(scored_mutations, key=lambda x: x['llr'], reverse=True)[:5]

    return {
        'n_total': len(scored_mutations),
        'n_deleterious': n_deleterious,
        'n_neutral': n_neutral,
        'pct_deleterious': round(n_deleterious / len(scored_mutations) * 100, 1),
        'llr_mean': round(llr_array.mean(), 4),
        'llr_std': round(llr_array.std(), 4),
        'llr_min': round(llr_array.min(), 4),
        'llr_max': round(llr_array.max(), 4),
        'top_deleterious': top_deleterious,
        'top_neutral': top_neutral,
    }