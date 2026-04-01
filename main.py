import os
import torch
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from model.cnn import SimpleCNN
from utils.data_loader import load_data
from utils.train import train_model
from methods.gradcam import get_gradcam_map
from methods.lime_explain import get_lime_map
from methods.integrated_gradients import get_integrated_gradients_map
from methods.shap_explain import get_shap_map

MODEL_PATH = "saved_models/simple_cnn.pth"

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def show_combined_explanations(model, image_tensor, true_label):
    image_np = image_tensor.permute(1, 2, 0).detach().numpy()

    # Grad-CAM
    gradcam_heatmap, gradcam_pred = get_gradcam_map(model, image_tensor)

    # LIME
    lime_map, lime_temp, lime_mask, lime_pred = get_lime_map(model, image_tensor)

    # Integrated Gradients
    ig_heatmap, ig_pred = get_integrated_gradients_map(model, image_tensor)

    # SHAP
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
    shap_plot = plt.imshow(shap_heatmap, cmap="jet")
    plt.title(f"SHAP Heatmap\nPred: {class_names[shap_pred]}")
    plt.axis("off")
    cbar = plt.colorbar(shap_plot, fraction=0.046, pad=0.04)
    cbar.set_label("Importance")

    plt.subplot(4, 3, 12)
    plt.imshow(image_np)
    plt.imshow(shap_heatmap, cmap="jet", alpha=0.5)
    plt.title("SHAP Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Step 1: Load dataset
    trainloader, testloader = load_data()

    # Step 2: Create CNN model
    model = SimpleCNN()

    # Make sure saved_models folder exists
    os.makedirs("saved_models", exist_ok=True)

    # Step 3: Ask user whether to train a new model or load saved model
    choice = input("Type 'T' to train a new model or 'L' to use saved model: (T/L) ").strip().lower()

    if choice == "t":
        print("Training new model...")
        model = train_model(model, trainloader)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    elif choice == "l":
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print(f"Loaded model from {MODEL_PATH}")

    else:
        print("Invalid choice. Please type 'T' or 'L'.")
        return

    # Step 4: Get one batch of test images
    images, labels = next(iter(testloader))

    # Show the next 3 images
    num_images = 3

    for i in range(num_images):
        print(f"\n--- Image {i + 1} ---")

        image_tensor = images[i]
        true_label = labels[i].item()

        print("Showing combined explanations...")
        show_combined_explanations(model, image_tensor, true_label)


if __name__ == "__main__":
    main()