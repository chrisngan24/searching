import numpy as np

def compute_precision_at_k(relevances, k):
    """
    Compute precision at rank k
    """
    trimmed_rel = relevances[:k]
    tp = sum(trimmed_rel == 1)
    return tp / float(k)

def compute_dcg(relevance, ranks,k):
    """
    Compute dcg@k
    """
    trimmed_rel = relevance[:k]
    trimmed_ranks = ranks[:k]
    return np.sum(np.divide(
        trimmed_rel,
        np.log2(trimmed_ranks + 1),
         ))
