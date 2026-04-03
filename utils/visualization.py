import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from methods.gradcam import get_gradcam_map
from methods.lime_explain import get_lime_map
from methods.integrated_gradients import get_integrated_gradients_map
from methods.shap_explain import get_shap_map


# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def show_combined_explanations(model, image_tensor, true_label):
    image_np = image_tensor.permute(1, 2, 0).detach().numpy()

    # =========================
    # Generate explanations
    # =========================
    gradcam_heatmap, gradcam_pred = get_gradcam_map(model, image_tensor)
    lime_map, lime_temp, lime_mask, lime_pred = get_lime_map(model, image_tensor)
    ig_heatmap, ig_pred = get_integrated_gradients_map(model, image_tensor)
    shap_heatmap, shap_pred = get_shap_map(model, image_tensor)

    plt.figure(figsize=(15, 16))

    # =========================
    # Row 1: Grad-CAM
    # =========================
    plt.subplot(4, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Grad-CAM Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    plt.subplot(4, 3, 2)
    gradcam_plot = plt.imshow(gradcam_heatmap, cmap="jet")
    plt.title(f"Grad-CAM Heatmap\nPred: {class_names[gradcam_pred]}")
    plt.axis("off")
    cbar = plt.colorbar(gradcam_plot, fraction=0.046, pad=0.04)
    cbar.set_label("Importance")

    plt.subplot(4, 3, 3)
    plt.imshow(image_np)
    plt.imshow(gradcam_heatmap, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")

    # =========================
    # Row 2: LIME
    # =========================
    plt.subplot(4, 3, 4)
    plt.imshow(image_np)
    plt.title(f"LIME Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    plt.subplot(4, 3, 5)
    plt.imshow(image_np)
    lime_plot = plt.imshow(lime_map, cmap="jet", alpha=0.5)
    plt.title(f"LIME Heatmap\nPred: {class_names[lime_pred]}")
    plt.axis("off")
    cbar = plt.colorbar(lime_plot, fraction=0.046, pad=0.04)
    cbar.set_label("Importance")

    plt.subplot(4, 3, 6)
    plt.imshow(mark_boundaries(image_np, lime_mask))
    plt.title("LIME Boundaries")
    plt.axis("off")

    # =========================
    # Row 3: Integrated Gradients
    # =========================
    plt.subplot(4, 3, 7)
    plt.imshow(image_np)
    plt.title(f"IG Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    plt.subplot(4, 3, 8)
    ig_plot = plt.imshow(ig_heatmap, cmap="jet")
    plt.title(f"IG Heatmap\nPred: {class_names[ig_pred]}")
    plt.axis("off")
    cbar = plt.colorbar(ig_plot, fraction=0.046, pad=0.04)
    cbar.set_label("Importance")

    plt.subplot(4, 3, 9)
    plt.imshow(image_np)
    plt.imshow(ig_heatmap, cmap="jet", alpha=0.5)
    plt.title("IG Overlay")
    plt.axis("off")

    # =========================
    # Row 4: SHAP
    # =========================
    plt.subplot(4, 3, 10)
    plt.imshow(image_np)
    plt.title(f"SHAP Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    plt.subplot(4, 3, 11)
    shap_plot = plt.imshow(shap_heatmap, cmap="bwr", vmin=-1, vmax=1)
    plt.title(f"SHAP Heatmap\nPred: {class_names[shap_pred]}")
    plt.axis("off")
    cbar = plt.colorbar(shap_plot, fraction=0.046, pad=0.04)
    cbar.set_label("Contribution")

    plt.subplot(4, 3, 12)
    plt.imshow(image_np)
    plt.imshow(shap_heatmap, cmap="bwr", vmin=-1, vmax=1, alpha=0.5)
    plt.title("SHAP Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()