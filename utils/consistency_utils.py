import numpy as np
from itertools import combinations


def cosine_similarity(map1, map2):
    # flatten maps into 1D vectors
    v1 = map1.flatten()
    v2 = map2.flatten()

    # avoid divide-by-zero
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(v1, v2) / (norm1 * norm2)


def average_pairwise_similarity(maps):
    # compare every pair of runs
    scores = []

    for map1, map2 in combinations(maps, 2):
        score = cosine_similarity(map1, map2)
        scores.append(score)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))