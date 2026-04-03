import os
import torch
import numpy as np
import random

from model.cnn import SimpleCNN
from utils.data_loader import load_data
from methods.gradcam import get_gradcam_map
from methods.lime_explain import get_lime_map
from methods.integrated_gradients import get_integrated_gradients_map
from methods.shap_explain import get_shap_map
from utils.consistency_utils import average_pairwise_similarity, average_pairwise_iou

MODEL_PATH = "saved_models/simple_cnn.pth"

# Change these values to test more or fewer images and runs per image
# More images are generally better for more reliable results compared to runs
# Keep in mind this scales as O(n^2), so 100 runs would be 10,000 comparisons per image
NUM_IMAGES = 30
NUM_RUNS = 10

# Top percentage of most important pixels used for IoU overlap
TOP_PERCENT = 0.10

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def safe_mean(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def format_score(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def collect_test_images(testloader, num_images):
    collected_images = []
    collected_labels = []

    for batch_images, batch_labels in testloader:
        for image, label in zip(batch_images, batch_labels):
            collected_images.append(image)
            collected_labels.append(label.item())

            if len(collected_images) == num_images:
                return collected_images, collected_labels

    return collected_images, collected_labels


def main():
    trainloader, testloader = load_data()

    model = SimpleCNN()

    if not os.path.exists(MODEL_PATH):
        print(f"Saved model not found at {MODEL_PATH}")
        print("Train the model first using main.py")
        return

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    images, labels = collect_test_images(testloader, NUM_IMAGES)

    if len(images) < NUM_IMAGES:
        print(f"Only found {len(images)} test images, not {NUM_IMAGES}")
        return

    # =========================
    # Cosine similarity scores
    # =========================
    gradcam_cosine_scores = []
    lime_cosine_scores = []
    ig_cosine_scores = []
    shap_cosine_scores = []

    gradcam_cosine_correct_scores = []
    lime_cosine_correct_scores = []
    ig_cosine_correct_scores = []
    shap_cosine_correct_scores = []

    gradcam_cosine_incorrect_scores = []
    lime_cosine_incorrect_scores = []
    ig_cosine_incorrect_scores = []
    shap_cosine_incorrect_scores = []

    # =========================
    # Top-k IoU scores
    # =========================
    gradcam_iou_scores = []
    lime_iou_scores = []
    ig_iou_scores = []
    shap_iou_scores = []

    gradcam_iou_correct_scores = []
    lime_iou_correct_scores = []
    ig_iou_correct_scores = []
    shap_iou_correct_scores = []

    gradcam_iou_incorrect_scores = []
    lime_iou_incorrect_scores = []
    ig_iou_incorrect_scores = []
    shap_iou_incorrect_scores = []

    correct_count = 0
    incorrect_count = 0

    for i in range(NUM_IMAGES):
        print("\n========================================")
        print(f"Testing Image {i + 1}/{NUM_IMAGES}")
        print("========================================")

        image_tensor = images[i]
        true_label = labels[i]

        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            predicted_label = torch.argmax(output, dim=1).item()

        is_correct = predicted_label == true_label

        print(f"True label:      {class_names[true_label]}")
        print(f"Predicted label: {class_names[predicted_label]}")
        print(f"Correct?         {is_correct}")

        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        # =========================
        # Grad-CAM
        # =========================
        gradcam_maps = []
        for run in range(NUM_RUNS):
            heatmap, _ = get_gradcam_map(model, image_tensor)
            gradcam_maps.append(heatmap)

        gradcam_cosine = average_pairwise_similarity(gradcam_maps)
        gradcam_iou = average_pairwise_iou(gradcam_maps, top_percent=TOP_PERCENT)

        gradcam_cosine_scores.append(gradcam_cosine)
        gradcam_iou_scores.append(gradcam_iou)

        print(f"Grad-CAM cosine consistency:      {gradcam_cosine:.4f}")
        print(f"Grad-CAM top-k IoU consistency:   {gradcam_iou:.4f}")

        if is_correct:
            gradcam_cosine_correct_scores.append(gradcam_cosine)
            gradcam_iou_correct_scores.append(gradcam_iou)
        else:
            gradcam_cosine_incorrect_scores.append(gradcam_cosine)
            gradcam_iou_incorrect_scores.append(gradcam_iou)

        # =========================
        # LIME
        # =========================
        lime_maps = []
        for run in range(NUM_RUNS):
            random.seed(None)
            np.random.seed(None)

            lime_map, _, _, _ = get_lime_map(model, image_tensor)
            lime_maps.append(lime_map)

        lime_cosine = average_pairwise_similarity(lime_maps)
        lime_iou = average_pairwise_iou(lime_maps, top_percent=TOP_PERCENT)

        lime_cosine_scores.append(lime_cosine)
        lime_iou_scores.append(lime_iou)

        print(f"LIME cosine consistency:          {lime_cosine:.4f}")
        print(f"LIME top-k IoU consistency:       {lime_iou:.4f}")

        if is_correct:
            lime_cosine_correct_scores.append(lime_cosine)
            lime_iou_correct_scores.append(lime_iou)
        else:
            lime_cosine_incorrect_scores.append(lime_cosine)
            lime_iou_incorrect_scores.append(lime_iou)

        # =========================
        # Integrated Gradients
        # =========================
        ig_maps = []
        for run in range(NUM_RUNS):
            heatmap, _ = get_integrated_gradients_map(model, image_tensor)
            ig_maps.append(heatmap)

        ig_cosine = average_pairwise_similarity(ig_maps)
        ig_iou = average_pairwise_iou(ig_maps, top_percent=TOP_PERCENT)

        ig_cosine_scores.append(ig_cosine)
        ig_iou_scores.append(ig_iou)

        print(f"Integrated Gradients cosine:      {ig_cosine:.4f}")
        print(f"Integrated Gradients top-k IoU:   {ig_iou:.4f}")

        if is_correct:
            ig_cosine_correct_scores.append(ig_cosine)
            ig_iou_correct_scores.append(ig_iou)
        else:
            ig_cosine_incorrect_scores.append(ig_cosine)
            ig_iou_incorrect_scores.append(ig_iou)

        # =========================
        # SHAP
        # =========================
        shap_maps = []
        for run in range(NUM_RUNS):
            random.seed(None)
            np.random.seed(None)

            heatmap, _ = get_shap_map(model, image_tensor)
            shap_maps.append(heatmap)

        shap_cosine = average_pairwise_similarity(shap_maps)
        shap_iou = average_pairwise_iou(shap_maps, top_percent=TOP_PERCENT)

        shap_cosine_scores.append(shap_cosine)
        shap_iou_scores.append(shap_iou)

        print(f"SHAP cosine consistency:          {shap_cosine:.4f}")
        print(f"SHAP top-k IoU consistency:       {shap_iou:.4f}")

        if is_correct:
            shap_cosine_correct_scores.append(shap_cosine)
            shap_iou_correct_scores.append(shap_iou)
        else:
            shap_cosine_incorrect_scores.append(shap_cosine)
            shap_iou_incorrect_scores.append(shap_iou)

    # =========================
    # Final averages - cosine
    # =========================
    overall_gradcam_cosine = safe_mean(gradcam_cosine_scores)
    overall_lime_cosine = safe_mean(lime_cosine_scores)
    overall_ig_cosine = safe_mean(ig_cosine_scores)
    overall_shap_cosine = safe_mean(shap_cosine_scores)

    correct_gradcam_cosine = safe_mean(gradcam_cosine_correct_scores)
    correct_lime_cosine = safe_mean(lime_cosine_correct_scores)
    correct_ig_cosine = safe_mean(ig_cosine_correct_scores)
    correct_shap_cosine = safe_mean(shap_cosine_correct_scores)

    incorrect_gradcam_cosine = safe_mean(gradcam_cosine_incorrect_scores)
    incorrect_lime_cosine = safe_mean(lime_cosine_incorrect_scores)
    incorrect_ig_cosine = safe_mean(ig_cosine_incorrect_scores)
    incorrect_shap_cosine = safe_mean(shap_cosine_incorrect_scores)

    # =========================
    # Final averages - IoU
    # =========================
    overall_gradcam_iou = safe_mean(gradcam_iou_scores)
    overall_lime_iou = safe_mean(lime_iou_scores)
    overall_ig_iou = safe_mean(ig_iou_scores)
    overall_shap_iou = safe_mean(shap_iou_scores)

    correct_gradcam_iou = safe_mean(gradcam_iou_correct_scores)
    correct_lime_iou = safe_mean(lime_iou_correct_scores)
    correct_ig_iou = safe_mean(ig_iou_correct_scores)
    correct_shap_iou = safe_mean(shap_iou_correct_scores)

    incorrect_gradcam_iou = safe_mean(gradcam_iou_incorrect_scores)
    incorrect_lime_iou = safe_mean(lime_iou_incorrect_scores)
    incorrect_ig_iou = safe_mean(ig_iou_incorrect_scores)
    incorrect_shap_iou = safe_mean(shap_iou_incorrect_scores)

    print("\n========================================")
    print("FINAL AVERAGE CONSISTENCY")
    print("========================================")
    print(f"Total images tested: {NUM_IMAGES}")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {incorrect_count}")
    print(f"Top-k percent used for IoU: {int(TOP_PERCENT * 100)}%")
    print("========================================")

    print("\nALL IMAGES - COSINE SIMILARITY")
    print(f"Grad-CAM:             {format_score(overall_gradcam_cosine)}")
    print(f"LIME:                 {format_score(overall_lime_cosine)}")
    print(f"Integrated Gradients: {format_score(overall_ig_cosine)}")
    print(f"SHAP:                 {format_score(overall_shap_cosine)}")

    print("\nALL IMAGES - TOP-K IoU")
    print(f"Grad-CAM:             {format_score(overall_gradcam_iou)}")
    print(f"LIME:                 {format_score(overall_lime_iou)}")
    print(f"Integrated Gradients: {format_score(overall_ig_iou)}")
    print(f"SHAP:                 {format_score(overall_shap_iou)}")

    print("\nCORRECT PREDICTIONS ONLY - COSINE SIMILARITY")
    print(f"Grad-CAM:             {format_score(correct_gradcam_cosine)}")
    print(f"LIME:                 {format_score(correct_lime_cosine)}")
    print(f"Integrated Gradients: {format_score(correct_ig_cosine)}")
    print(f"SHAP:                 {format_score(correct_shap_cosine)}")

    print("\nCORRECT PREDICTIONS ONLY - TOP-K IoU")
    print(f"Grad-CAM:             {format_score(correct_gradcam_iou)}")
    print(f"LIME:                 {format_score(correct_lime_iou)}")
    print(f"Integrated Gradients: {format_score(correct_ig_iou)}")
    print(f"SHAP:                 {format_score(correct_shap_iou)}")

    print("\nINCORRECT PREDICTIONS ONLY - COSINE SIMILARITY")
    print(f"Grad-CAM:             {format_score(incorrect_gradcam_cosine)}")
    print(f"LIME:                 {format_score(incorrect_lime_cosine)}")
    print(f"Integrated Gradients: {format_score(incorrect_ig_cosine)}")
    print(f"SHAP:                 {format_score(incorrect_shap_cosine)}")

    print("\nINCORRECT PREDICTIONS ONLY - TOP-K IoU")
    print(f"Grad-CAM:             {format_score(incorrect_gradcam_iou)}")
    print(f"LIME:                 {format_score(incorrect_lime_iou)}")
    print(f"Integrated Gradients: {format_score(incorrect_ig_iou)}")
    print(f"SHAP:                 {format_score(incorrect_shap_iou)}")
    print("========================================")


if __name__ == "__main__":
    main()