# Format a score for display, returning "N/A" when the value is missing
def format_score(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


# Print consistency metrics for one image across all explanation methods
def print_image_results(
    image_num,
    total_images,
    true_label,
    predicted_label,
    is_correct,
    gradcam_cosine,
    lime_cosine,
    ig_cosine,
    shap_cosine,
    gradcam_iou,
    lime_iou,
    ig_iou,
    shap_iou
):
    # Show which image is being analyzed and whether the model predicted correctly
    print("\n========================================")
    print(
        f"Image {image_num}/{total_images} | "
        f"True: {true_label} | "
        f"Pred: {predicted_label} | "
        f"Correct: {is_correct}"
    )
    print("----------------------------------------")
    print("Method        Cosine   IoU")
    print("----------------------------------------")
    print(f"Grad-CAM      {gradcam_cosine:.3f}    {gradcam_iou:.3f}")
    print(f"LIME          {lime_cosine:.3f}    {lime_iou:.3f}")
    print(f"IG            {ig_cosine:.3f}    {ig_iou:.3f}")
    print(f"SHAP          {shap_cosine:.3f}    {shap_iou:.3f}")


# Print the final averaged consistency metrics for the full test set
def print_final_summary(
    num_images,
    correct_count,
    incorrect_count,
    top_percent,
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
    incorrect_shap_iou
):
    print("\n========================================")
    print("FINAL AVERAGE CONSISTENCY")
    print("========================================")
    print(f"Total images tested: {num_images}")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {incorrect_count}")
    print(f"Top-k percent used for IoU: {int(top_percent * 100)}%")
    print("========================================")

    # Print overall consistency scores across the full image set
    print("\nALL IMAGES - COSINE SIMILARITY")
    print(f"Grad-CAM:             {format_score(overall_gradcam_cosine)}")
    print(f"LIME:                 {format_score(overall_lime_cosine)}")
    print(f"Integrated Gradients: {format_score(overall_ig_cosine)}")
    print(f"SHAP:                 {format_score(overall_shap_cosine)}")

    # Print overall IoU scores across the full image set
    print("\nALL IMAGES - TOP-K IoU")
    print(f"Grad-CAM:             {format_score(overall_gradcam_iou)}")
    print(f"LIME:                 {format_score(overall_lime_iou)}")
    print(f"Integrated Gradients: {format_score(overall_ig_iou)}")
    print(f"SHAP:                 {format_score(overall_shap_iou)}")

    # Print consistency scores only for images the model predicted correctly
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

    # Print consistency scores only for images the model predicted incorrectly
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