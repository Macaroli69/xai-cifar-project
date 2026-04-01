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


def explain_with_shap(model, image_tensor, true_label):
    model.eval()

    # Convert tensor image to numpy image for SHAP
    # PyTorch: (C, H, W) -> SHAP: (H, W, C)
    image_np = image_tensor.permute(1, 2, 0).numpy()

    # Add batch dimension for direct model prediction
    input_tensor = image_tensor.unsqueeze(0)

    # Prediction function for SHAP
    # SHAP gives images in shape (N, H, W, C)
    # model needs (N, C, H, W)
    def predict_fn(images_batch):
        images_batch = torch.tensor(images_batch, dtype=torch.float32)
        images_batch = images_batch.permute(0, 3, 1, 2)

        with torch.no_grad():
            outputs = model(images_batch)
            probs = torch.softmax(outputs, dim=1)

        return probs.numpy()

    # Get model prediction for this image
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    # SHAP image masker
    masker = shap.maskers.Image("blur(8,8)", image_np.shape)

    # Create explainer
    explainer = shap.Explainer(predict_fn, masker)

    # Explain one image
    # max_evals kept lower so runtime is manageable
    shap_values = explainer(
        image_np[np.newaxis, ...],
        max_evals=100,
        batch_size=50
    )

    # SHAP values for the predicted class
    values = shap_values.values[0, :, :, :, predicted_label]

    # Convert to a simpler heatmap by averaging color channels
    heatmap = np.mean(np.abs(values), axis=2)

    # Normalize heatmap
    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    # Plot results
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