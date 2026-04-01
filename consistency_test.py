import os
import torch
import numpy as np
import random

from model.cnn import SimpleCNN
from utils.data_loader import load_data
from methods.gradcam import get_gradcam_map
from methods.lime_explain import get_lime_map
from methods.shap_explain import get_shap_map
from utils.consistency_utils import average_pairwise_similarity

MODEL_PATH = "saved_models/simple_cnn.pth"

NUM_IMAGES = 30
NUM_RUNS = 10

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


def main():
    # load data
    trainloader, testloader = load_data()

    # load model
    model = SimpleCNN()

    if not os.path.exists(MODEL_PATH):
        print(f"Saved model not found at {MODEL_PATH}")
        print("Train the model first using main.py")
        return

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # get test images
    images, labels = next(iter(testloader))

    # overall scores
    gradcam_scores = []
    lime_scores = []
    shap_scores = []

    # correct-only scores
    gradcam_correct_scores = []
    lime_correct_scores = []
    shap_correct_scores = []

    # incorrect-only scores
    gradcam_incorrect_scores = []
    lime_incorrect_scores = []
    shap_incorrect_scores = []

    correct_count = 0
    incorrect_count = 0

    for i in range(NUM_IMAGES):
        print("\n==============================")
        print(f"Testing Image {i + 1}/{NUM_IMAGES}")
        print("==============================")

        image_tensor = images[i]
        true_label = labels[i].item()

        # get model prediction once for correctness check
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

        # -------------------------
        # Grad-CAM (deterministic)
        # -------------------------
        gradcam_maps = []
        for run in range(NUM_RUNS):
            heatmap, _ = get_gradcam_map(model, image_tensor)
            gradcam_maps.append(heatmap)

        gradcam_score = average_pairwise_similarity(gradcam_maps)
        gradcam_scores.append(gradcam_score)
        print(f"Grad-CAM consistency: {gradcam_score:.4f}")

        if is_correct:
            gradcam_correct_scores.append(gradcam_score)
        else:
            gradcam_incorrect_scores.append(gradcam_score)

        # -------------------------
        # LIME (random)
        # -------------------------
        lime_maps = []
        for run in range(NUM_RUNS):
            random.seed(None)
            np.random.seed(None)

            lime_map, _, _, _ = get_lime_map(model, image_tensor)
            lime_maps.append(lime_map)

        lime_score = average_pairwise_similarity(lime_maps)
        lime_scores.append(lime_score)
        print(f"LIME consistency:     {lime_score:.4f}")

        if is_correct:
            lime_correct_scores.append(lime_score)
        else:
            lime_incorrect_scores.append(lime_score)

        # -------------------------
        # SHAP (semi-random)
        # -------------------------
        shap_maps = []
        for run in range(NUM_RUNS):
            random.seed(None)
            np.random.seed(None)

            shap_map, _ = get_shap_map(model, image_tensor)
            shap_maps.append(shap_map)

        shap_score = average_pairwise_similarity(shap_maps)
        shap_scores.append(shap_score)
        print(f"SHAP consistency:     {shap_score:.4f}")

        if is_correct:
            shap_correct_scores.append(shap_score)
        else:
            shap_incorrect_scores.append(shap_score)

    # -------------------------
    # Final Results
    # -------------------------
    overall_gradcam = safe_mean(gradcam_scores)
    overall_lime = safe_mean(lime_scores)
    overall_shap = safe_mean(shap_scores)

    correct_gradcam = safe_mean(gradcam_correct_scores)
    correct_lime = safe_mean(lime_correct_scores)
    correct_shap = safe_mean(shap_correct_scores)

    incorrect_gradcam = safe_mean(gradcam_incorrect_scores)
    incorrect_lime = safe_mean(lime_incorrect_scores)
    incorrect_shap = safe_mean(shap_incorrect_scores)

    print("\n========================================")
    print("FINAL AVERAGE CONSISTENCY")
    print("========================================")
    print(f"Total images tested: {NUM_IMAGES}")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {incorrect_count}")
    print("========================================")

    print("\nALL IMAGES")
    print(f"Grad-CAM: {format_score(overall_gradcam)}")
    print(f"LIME:     {format_score(overall_lime)}")
    print(f"SHAP:     {format_score(overall_shap)}")

    print("\nCORRECT PREDICTIONS ONLY")
    print(f"Grad-CAM: {format_score(correct_gradcam)}")
    print(f"LIME:     {format_score(correct_lime)}")
    print(f"SHAP:     {format_score(correct_shap)}")

    print("\nINCORRECT PREDICTIONS ONLY")
    print(f"Grad-CAM: {format_score(incorrect_gradcam)}")
    print(f"LIME:     {format_score(incorrect_lime)}")
    print(f"SHAP:     {format_score(incorrect_shap)}")
    print("========================================")


if __name__ == "__main__":
    main()