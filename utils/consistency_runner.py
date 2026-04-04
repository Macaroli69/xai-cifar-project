import io
import random
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

from methods.gradcam import get_gradcam_map
from methods.lime_explain import get_lime_map
from methods.integrated_gradients import get_integrated_gradients_map
from methods.shap_explain import get_shap_map
from utils.consistency_utils import average_pairwise_similarity, average_pairwise_iou


# Calculate mean of values, or return None if the list is empty
def safe_mean(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


# Extract the first N images and their labels from a test dataloader
def collect_test_images(testloader, num_images):
    collected_images = []
    collected_labels = []

    # Iterate through batches and collect images until reaching the specified count
    for batch_images, batch_labels in testloader:
        for image, label in zip(batch_images, batch_labels):
            collected_images.append(image)
            collected_labels.append(label.item())

            if len(collected_images) == num_images:
                return collected_images, collected_labels

    return collected_images, collected_labels


# Execute a function while suppressing its print statements and error messages
def run_with_suppressed_output(func):
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        return func()


# Calculate consistency metrics (cosine similarity and IoU) for a set of explanation maps
def calculate_consistency_scores(maps, top_percent):
    cosine_score = average_pairwise_similarity(maps)
    iou_score = average_pairwise_iou(maps, top_percent=top_percent)
    return cosine_score, iou_score


# Run Grad-CAM multiple times on the same image and measure consistency of explanations
def run_gradcam_consistency(model, image_tensor, num_runs, top_percent):
    method_maps = []

    # Generate multiple Grad-CAM explanations for the same image
    for _ in range(num_runs):
        heatmap, _ = get_gradcam_map(model, image_tensor)
        method_maps.append(heatmap)

    return calculate_consistency_scores(method_maps, top_percent)


# Run LIME multiple times on the same image and measure consistency of explanations
def run_lime_consistency(model, image_tensor, num_runs, top_percent):
    method_maps = []

    # Generate multiple LIME explanations for the same image
    # Reset random seeds each iteration to allow LIME's stochastic sampling to vary
    for _ in range(num_runs):
        random.seed(None)
        np.random.seed(None)

        lime_map, _, _, _ = run_with_suppressed_output(
            lambda: get_lime_map(model, image_tensor)
        )
        method_maps.append(lime_map)

    return calculate_consistency_scores(method_maps, top_percent)


# Run Integrated Gradients multiple times on the same image and measure consistency of explanations
def run_ig_consistency(model, image_tensor, num_runs, top_percent):
    method_maps = []

    # Generate multiple IG explanations for the same image
    for _ in range(num_runs):
        heatmap, _ = get_integrated_gradients_map(model, image_tensor)
        method_maps.append(heatmap)

    return calculate_consistency_scores(method_maps, top_percent)


# Run SHAP multiple times on the same image and measure consistency of explanations
def run_shap_consistency(model, image_tensor, num_runs, top_percent):
    method_maps = []

    # Generate multiple SHAP explanations for the same image
    # Reset random seeds each iteration to allow SHAP's stochastic sampling to vary
    for _ in range(num_runs):
        random.seed(None)
        np.random.seed(None)

        heatmap, _ = run_with_suppressed_output(
            lambda: get_shap_map(model, image_tensor)
        )
        method_maps.append(heatmap)

    return calculate_consistency_scores(method_maps, top_percent)