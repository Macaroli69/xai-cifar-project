import numpy as np
from itertools import combinations


def cosine_similarity(map1, map2):
    # Flatten maps into 1D vectors
    v1 = map1.flatten()
    v2 = map2.flatten()

    # Avoid divide-by-zero
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def average_pairwise_similarity(maps):
    # Compare every pair of runs using cosine similarity
    scores = []

    for map1, map2 in combinations(maps, 2):
        score = cosine_similarity(map1, map2)
        scores.append(score)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))


def top_k_mask(heatmap, top_percent=0.10):
    # Flatten heatmap
    flat = heatmap.flatten()

    # Number of pixels to keep
    k = max(1, int(len(flat) * top_percent))

    # Find threshold for top-k values
    top_indices = np.argpartition(flat, -k)[-k:]

    # Create binary mask
    mask = np.zeros_like(flat, dtype=np.uint8)
    mask[top_indices] = 1

    return mask.reshape(heatmap.shape)


def iou_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def average_pairwise_iou(maps, top_percent=0.10):
    # Compare every pair of runs using top-k IoU
    scores = []

    for map1, map2 in combinations(maps, 2):
        mask1 = top_k_mask(map1, top_percent=top_percent)
        mask2 = top_k_mask(map2, top_percent=top_percent)
        score = iou_score(mask1, mask2)
        scores.append(score)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))