# WHAT SHAP IS DOING:
# 1. Take one test image
# 2. Mask different parts of the image
# 3. See how the model prediction changes
# 4. Estimate how much each region contributes
# 5. Show which parts of the image matter most

import numpy as np
import torch
import matplotlib.pyplot as plt
import shap


# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_shap_map(model, image_tensor):
    model.eval()

    # Convert tensor image to numpy image
    image_np = image_tensor.permute(1, 2, 0).numpy()

    # Add batch dimension for direct model prediction
    input_tensor = image_tensor.unsqueeze(0)

    # Prediction function for SHAP
    def predict_fn(images_batch):
        images_batch = torch.tensor(images_batch, dtype=torch.float32)
        images_batch = images_batch.permute(0, 3, 1, 2)

        with torch.no_grad():
            outputs = model(images_batch)
            probs = torch.softmax(outputs, dim=1)

        return probs.numpy()

    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    # SHAP masker
    masker = shap.maskers.Image("blur(8,8)", image_np.shape)

    # Create SHAP explainer
    explainer = shap.Explainer(predict_fn, masker)

    # Explain one image
    shap_values = explainer(
        image_np[np.newaxis, ...],
        max_evals=100,
        batch_size=50
    )

    # Get SHAP values for predicted class
    values = shap_values.values[0, :, :, :, predicted_label]

    # Turn 3-channel SHAP values into 2D heatmap
    heatmap = np.mean(np.abs(values), axis=2)

    # Normalize heatmap
    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    return heatmap, predicted_label


def explain_with_shap(model, image_tensor, true_label):
    heatmap, predicted_label = get_shap_map(model, image_tensor)

    image_np = image_tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    # SHAP heatmap
    plt.subplot(1, 3, 2)
    shap_plot = plt.imshow(heatmap, cmap="jet")
    plt.title(f"SHAP Heatmap\nPred: {class_names[predicted_label]}")
    plt.axis("off")

    cbar = plt.colorbar(shap_plot, fraction=0.046, pad=0.04)
    cbar.set_label("Importance")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image_np)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()