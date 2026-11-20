from typing import Any, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

encoder_model = SentenceTransformer("all-MiniLM-L6-v2")

def text2list(rubric):
    if rubric != "":
     return rubric.split("\n\n")
    return

def unbalanced_sinkhorn_distance(
    similarity: np.ndarray, 
    epsilon: float = 0.05, 
    tau: float = 0.5, 
    num_iters: int = 50
) -> np.ndarray:
    """
    Computes Unbalanced Sinkhorn transport matrix P using KL-relaxed scaling updates.
    Prevents artificial 1-to-1 mass splitting on unequal rubric dimensions (N != M).
    """
    cost_matrix = 1.0 - similarity
    gibbs_kernel = np.exp(-cost_matrix / epsilon)

    rows, cols = gibbs_kernel.shape
    a = np.ones((rows, 1)) / rows
    b = np.ones((cols, 1)) / cols

    u = np.ones((rows, 1))
    v = np.ones((cols, 1))
    eps_stable = 1e-12

    # Unbalanced exponent adjustment: fi = tau / (tau + epsilon)
    fi = tau / (tau + epsilon)

    for _ in range(num_iters):
        v = (b / (gibbs_kernel.T @ u + eps_stable)) ** fi
        u = (a / (gibbs_kernel @ v + eps_stable)) ** fi

    # Unbalanced Transport Matrix P [num_old, num_new]
    P = np.diag(u.squeeze()) @ gibbs_kernel @ np.diag(v.squeeze())
    return P
def get_dimensions_embedding(rubric_units, encoder) -> np.ndarray:
    """Encodes rubric dimension strings into L2-normalized numpy embeddings."""
    if not rubric_units:
        return np.empty((0, 0))

    embeddings = encoder.encode(rubric_units, normalize_embeddings=True)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    return embeddings

def get_score_novelty(new_rubric, rubric, eps, tau):
    if not rubric or not new_rubric:
            return False
    
    new_rubric_units = text2list(new_rubric)
    rubric_units = text2list(rubric)
            
        
    print(f"Num dimensions in new rurbic: {len(new_rubric_units)}")
    print(f"Num dimensions in current rubric: {len(rubric_units)}")
    
    if not rubric_units or not new_rubric_units:
        print("Error in parsing dimensions")
        return False
    
    rubric_emb = get_dimensions_embedding(rubric_units, encoder_model)
    new_rubric_emb = get_dimensions_embedding(new_rubric_units, encoder_model)
    
    similarity = rubric_emb @ new_rubric_emb.T
    num_old, num_new = similarity.shape
    
    # Compute Unbalanced Transport Matrix
    P_raw = unbalanced_sinkhorn_distance(similarity, epsilon=0.05, tau=tau, num_iters=20)
            
    # Renormalize P_raw into a valid joint probability distribution P_joint
    total_mass = np.sum(P_raw) + eps
    p_joint = P_raw / total_mass  # Sum of p_joint entries equals 1.0
    
    # Compute Marginals
    p_rubric = np.sum(p_joint, axis=1)   # P(R)   [num_old]
    
    # P(R_t | R) = P(R, R_t) / P(R)
    p_new_rubric_given_rubric = p_joint / (np.sum(p_joint, axis=1, keepdims=True) + eps)
    novelty_entropy_per_row = -np.sum(p_new_rubric_given_rubric * np.log2(p_new_rubric_given_rubric + eps), axis=1)
    global_novelty_h = np.sum(p_rubric * novelty_entropy_per_row)
    
    # Normalize via Maximum Theoretical Bounds (Log2 N and Log2 M)
    max_h_new = np.log2(num_new) if num_new > 1 else 1.0
    score_novelty = float(global_novelty_h / max_h_new)
    
    return score_novelty

def is_drifted(new_rubric, initial_rubric, threshold = 0.75, eps = 1e-12, tau = 0.5):
    score_novelty = get_score_novelty(new_rubric, initial_rubric, eps, tau)
    return bool(score_novelty > threshold)

def is_novel(new_rubric, candidate_rubric, threshold = 0.15, eps = 1e-12, tau = 0.5):    
    score_novelty = get_score_novelty(new_rubric, candidate_rubric, eps, tau)
    
    return bool(score_novelty > threshold)