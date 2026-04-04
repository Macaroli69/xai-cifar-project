import os
import torch

from model.cnn import SimpleCNN
from utils.data_loader import load_data
from utils.consistency_runner import (
    collect_test_images,
    run_gradcam_consistency,
    run_lime_consistency,
    run_ig_consistency,
    run_shap_consistency,
    safe_mean
)
from utils.results_saver import save_summary_csv, save_detailed_csv
from utils.consistency_display import print_image_results, print_final_summary

MODEL_PATH = "saved_models/simple_cnn.pth"

# Change these values to test more or fewer images and runs per image
NUM_IMAGES = 100
NUM_RUNS = 7

# Top percentage of most important pixels used for IoU overlap
TOP_PERCENT = 0.10

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def main():
    print(f"Running Consistency Test")
    print(f"Using {NUM_IMAGES} Images. And {NUM_RUNS} Runs per Image.")

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

    detailed_results = []

    # All images
    gradcam_cosine_scores = []
    lime_cosine_scores = []
    ig_cosine_scores = []
    shap_cosine_scores = []

    gradcam_iou_scores = []
    lime_iou_scores = []
    ig_iou_scores = []
    shap_iou_scores = []

    # Correct only
    gradcam_cosine_correct_scores = []
    lime_cosine_correct_scores = []
    ig_cosine_correct_scores = []
    shap_cosine_correct_scores = []

    gradcam_iou_correct_scores = []
    lime_iou_correct_scores = []
    ig_iou_correct_scores = []
    shap_iou_correct_scores = []

    # Incorrect only
    gradcam_cosine_incorrect_scores = []
    lime_cosine_incorrect_scores = []
    ig_cosine_incorrect_scores = []
    shap_cosine_incorrect_scores = []

    gradcam_iou_incorrect_scores = []
    lime_iou_incorrect_scores = []
    ig_iou_incorrect_scores = []
    shap_iou_incorrect_scores = []

    correct_count = 0
    incorrect_count = 0

    for i in range(NUM_IMAGES):
        image_tensor = images[i]
        true_label = labels[i]

        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            predicted_label = torch.argmax(output, dim=1).item()

        is_correct = predicted_label == true_label

        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        gradcam_cosine, gradcam_iou = run_gradcam_consistency(
            model, image_tensor, NUM_RUNS, TOP_PERCENT
        )
        lime_cosine, lime_iou = run_lime_consistency(
            model, image_tensor, NUM_RUNS, TOP_PERCENT
        )
        ig_cosine, ig_iou = run_ig_consistency(
            model, image_tensor, NUM_RUNS, TOP_PERCENT
        )
        shap_cosine, shap_iou = run_shap_consistency(
            model, image_tensor, NUM_RUNS, TOP_PERCENT
        )

        # Save all-image scores
        gradcam_cosine_scores.append(gradcam_cosine)
        lime_cosine_scores.append(lime_cosine)
        ig_cosine_scores.append(ig_cosine)
        shap_cosine_scores.append(shap_cosine)

        gradcam_iou_scores.append(gradcam_iou)
        lime_iou_scores.append(lime_iou)
        ig_iou_scores.append(ig_iou)
        shap_iou_scores.append(shap_iou)

        # Save correct / incorrect groups
        if is_correct:
            gradcam_cosine_correct_scores.append(gradcam_cosine)
            lime_cosine_correct_scores.append(lime_cosine)
            ig_cosine_correct_scores.append(ig_cosine)
            shap_cosine_correct_scores.append(shap_cosine)

            gradcam_iou_correct_scores.append(gradcam_iou)
            lime_iou_correct_scores.append(lime_iou)
            ig_iou_correct_scores.append(ig_iou)
            shap_iou_correct_scores.append(shap_iou)
        else:
            gradcam_cosine_incorrect_scores.append(gradcam_cosine)
            lime_cosine_incorrect_scores.append(lime_cosine)
            ig_cosine_incorrect_scores.append(ig_cosine)
            shap_cosine_incorrect_scores.append(shap_cosine)

            gradcam_iou_incorrect_scores.append(gradcam_iou)
            lime_iou_incorrect_scores.append(lime_iou)
            ig_iou_incorrect_scores.append(ig_iou)
            shap_iou_incorrect_scores.append(shap_iou)

        detailed_results.append([
            i + 1,
            class_names[true_label],
            class_names[predicted_label],
            is_correct,
            gradcam_cosine,
            lime_cosine,
            ig_cosine,
            shap_cosine,
            gradcam_iou,
            lime_iou,
            ig_iou,
            shap_iou
        ])

        print_image_results(
            image_num=i + 1,
            total_images=NUM_IMAGES,
            true_label=class_names[true_label],
            predicted_label=class_names[predicted_label],
            is_correct=is_correct,
            gradcam_cosine=gradcam_cosine,
            lime_cosine=lime_cosine,
            ig_cosine=ig_cosine,
            shap_cosine=shap_cosine,
            gradcam_iou=gradcam_iou,
            lime_iou=lime_iou,
            ig_iou=ig_iou,
            shap_iou=shap_iou
        )

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

    print_final_summary(
        num_images=NUM_IMAGES,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        top_percent=TOP_PERCENT,
        overall_gradcam_cosine=overall_gradcam_cosine,
        overall_lime_cosine=overall_lime_cosine,
        overall_ig_cosine=overall_ig_cosine,
        overall_shap_cosine=overall_shap_cosine,
        overall_gradcam_iou=overall_gradcam_iou,
        overall_lime_iou=overall_lime_iou,
        overall_ig_iou=overall_ig_iou,
        overall_shap_iou=overall_shap_iou,
        correct_gradcam_cosine=correct_gradcam_cosine,
        correct_lime_cosine=correct_lime_cosine,
        correct_ig_cosine=correct_ig_cosine,
        correct_shap_cosine=correct_shap_cosine,
        correct_gradcam_iou=correct_gradcam_iou,
        correct_lime_iou=correct_lime_iou,
        correct_ig_iou=correct_ig_iou,
        correct_shap_iou=correct_shap_iou,
        incorrect_gradcam_cosine=incorrect_gradcam_cosine,
        incorrect_lime_cosine=incorrect_lime_cosine,
        incorrect_ig_cosine=incorrect_ig_cosine,
        incorrect_shap_cosine=incorrect_shap_cosine,
        incorrect_gradcam_iou=incorrect_gradcam_iou,
        incorrect_lime_iou=incorrect_lime_iou,
        incorrect_ig_iou=incorrect_ig_iou,
        incorrect_shap_iou=incorrect_shap_iou
    )

    save_choice = input("\nSave results to CSV? (y/n): ").strip().lower()

    if save_choice == "y":
        summary_filename = save_summary_csv(
            num_images=NUM_IMAGES,
            num_runs=NUM_RUNS,
            overall_gradcam_cosine=overall_gradcam_cosine,
            overall_lime_cosine=overall_lime_cosine,
            overall_ig_cosine=overall_ig_cosine,
            overall_shap_cosine=overall_shap_cosine,
            overall_gradcam_iou=overall_gradcam_iou,
            overall_lime_iou=overall_lime_iou,
            overall_ig_iou=overall_ig_iou,
            overall_shap_iou=overall_shap_iou,
            correct_gradcam_cosine=correct_gradcam_cosine,
            correct_lime_cosine=correct_lime_cosine,
            correct_ig_cosine=correct_ig_cosine,
            correct_shap_cosine=correct_shap_cosine,
            correct_gradcam_iou=correct_gradcam_iou,
            correct_lime_iou=correct_lime_iou,
            correct_ig_iou=correct_ig_iou,
            correct_shap_iou=correct_shap_iou,
            incorrect_gradcam_cosine=incorrect_gradcam_cosine,
            incorrect_lime_cosine=incorrect_lime_cosine,
            incorrect_ig_cosine=incorrect_ig_cosine,
            incorrect_shap_cosine=incorrect_shap_cosine,
            incorrect_gradcam_iou=incorrect_gradcam_iou,
            incorrect_lime_iou=incorrect_lime_iou,
            incorrect_ig_iou=incorrect_ig_iou,
            incorrect_shap_iou=incorrect_shap_iou,
            correct_count=correct_count,
            incorrect_count=incorrect_count
        )

        detailed_filename = save_detailed_csv(
            num_images=NUM_IMAGES,
            num_runs=NUM_RUNS,
            detailed_results=detailed_results
        )

        print(f"\nSaved summary results to: {summary_filename}")
        print(f"Saved detailed results to: {detailed_filename}")
    else:
        print("Results were not saved.")


if __name__ == "__main__":
    main()