import csv
import os
from datetime import datetime


# Build a folder path for results based on image and run counts
def get_results_dir(num_images, num_runs):
    return os.path.join("results", f"{num_images}_images_{num_runs}_runs")


# Create the results directory if it does not already exist
def ensure_results_dir(num_images, num_runs):
    results_dir = get_results_dir(num_images, num_runs)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    return results_dir


# Save aggregated consistency metrics into a timestamped CSV file
def save_summary_csv(
    num_images,
    num_runs,
    overall_gradcam_cosine,
    overall_lime_cosine,
    overall_ig_cosine,
    overall_shap_cosine,
    overall_gradcam_iou,
    overall_lime_iou,
    overall_ig_iou,
    overall_shap_iou,
    correct_gradcam_cosine,
    correct_lime_cosine,
    correct_ig_cosine,
    correct_shap_cosine,
    correct_gradcam_iou,
    correct_lime_iou,
    correct_ig_iou,
    correct_shap_iou,
    incorrect_gradcam_cosine,
    incorrect_lime_cosine,
    incorrect_ig_cosine,
    incorrect_shap_cosine,
    incorrect_gradcam_iou,
    incorrect_lime_iou,
    incorrect_ig_iou,
    incorrect_shap_iou,
    correct_count,
    incorrect_count
):
    # Ensure the output folder exists before writing the CSV file
    results_dir = ensure_results_dir(num_images, num_runs)

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    filename = os.path.join(results_dir, f"consistency_summary_{timestamp}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Header row describes the grouping and metrics columns
        writer.writerow(["group", "method", "cosine_similarity", "top_k_iou"])

        # Save overall metrics for all images
        writer.writerow(["all_images", "Grad-CAM", overall_gradcam_cosine, overall_gradcam_iou])
        writer.writerow(["all_images", "LIME", overall_lime_cosine, overall_lime_iou])
        writer.writerow(["all_images", "Integrated Gradients", overall_ig_cosine, overall_ig_iou])
        writer.writerow(["all_images", "SHAP", overall_shap_cosine, overall_shap_iou])

        # Save metrics only for correctly predicted images if they exist
        if correct_count > 0:
            writer.writerow(["correct_only", "Grad-CAM", correct_gradcam_cosine, correct_gradcam_iou])
            writer.writerow(["correct_only", "LIME", correct_lime_cosine, correct_lime_iou])
            writer.writerow(["correct_only", "Integrated Gradients", correct_ig_cosine, correct_ig_iou])
            writer.writerow(["correct_only", "SHAP", correct_shap_cosine, correct_shap_iou])

        # Save metrics only for incorrectly predicted images if they exist
        if incorrect_count > 0:
            writer.writerow(["incorrect_only", "Grad-CAM", incorrect_gradcam_cosine, incorrect_gradcam_iou])
            writer.writerow(["incorrect_only", "LIME", incorrect_lime_cosine, incorrect_lime_iou])
            writer.writerow(["incorrect_only", "Integrated Gradients", incorrect_ig_cosine, incorrect_ig_iou])
            writer.writerow(["incorrect_only", "SHAP", incorrect_shap_cosine, incorrect_shap_iou])

    return filename


# Save detailed per-image consistency values to a timestamped CSV file
def save_detailed_csv(num_images, num_runs, detailed_results):
    results_dir = ensure_results_dir(num_images, num_runs)

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    filename = os.path.join(results_dir, f"consistency_detailed_{timestamp}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header row describing each column in the detailed results
        writer.writerow([
            "image_index",
            "true_label",
            "predicted_label",
            "is_correct",
            "gradcam_cosine",
            "lime_cosine",
            "ig_cosine",
            "shap_cosine",
            "gradcam_iou",
            "lime_iou",
            "ig_iou",
            "shap_iou"
        ])
        # Write one row per image with all stored metrics
        writer.writerows(detailed_results)

    return filename