# WHAT SHAP IS DOING:
# 1. Take one test image
# 2. Mask different parts of the image
# 3. See how the model prediction changes
# 4. Estimate how much each region contributes
# 5. Show which parts of the image matter most
#
# IMPORTANT NOTE:
# This version keeps the sign of SHAP values.
# Positive values support the predicted class.
# Negative values go against the predicted class.
#
# To introduce variation across runs, this version slightly randomizes:
# - the blur masker size
# - the SHAP evaluation budget
#
# This keeps SHAP closer to the image-based explanation style that fits
# this project better, while avoiding the instability from flattening the
# full image into thousands of raw features for SamplingExplainer.

import random
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
    image_np = image_tensor.permute(1, 2, 0).detach().numpy().astype(np.float32)

    # Add batch dimension for direct model prediction
    input_tensor = image_tensor.unsqueeze(0)

    # Prediction function for SHAP
    def predict_fn(images_batch):
        images_batch = np.array(images_batch, dtype=np.float32)
        images_batch = torch.tensor(images_batch, dtype=torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            outputs = model(images_batch)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    # Randomize blur size slightly to introduce variation across runs
    blur_size = random.choice([4, 6, 8])

    # Randomize evaluation budget slightly to introduce variation across runs
    max_evals = random.choice([80, 100, 120])

    # SHAP masker
    masker = shap.maskers.Image(f"blur({blur_size},{blur_size})", image_np.shape)

    # Create SHAP explainer
    explainer = shap.Explainer(predict_fn, masker)

    # Explain one image
    shap_values = explainer(
        image_np[np.newaxis, ...],
        max_evals=max_evals,
        batch_size=50
    )

    # SHAP output shape for image explanations is typically:
    # (batch, height, width, channels, classes)
    values = shap_values.values[0, :, :, :, predicted_label]

    # Keep sign information:
    # positive = supports predicted class
    # negative = goes against predicted class
    heatmap = np.mean(values, axis=2)

    # Normalize to range [-1, 1]
    max_abs = np.max(np.abs(heatmap))
    if max_abs != 0:
        heatmap = heatmap / max_abs

    return heatmap, predicted_label


def explain_with_shap(model, image_tensor, true_label):
    heatmap, predicted_label = get_shap_map(model, image_tensor)

    image_np = image_tensor.permute(1, 2, 0).detach().numpy()

    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    # SHAP signed heatmap
    plt.subplot(1, 3, 2)
    shap_plot = plt.imshow(heatmap, cmap="bwr", vmin=-1, vmax=1)
    plt.title(f"SHAP Heatmap\nPred: {class_names[predicted_label]}")
    plt.axis("off")

    cbar = plt.colorbar(shap_plot, fraction=0.046, pad=0.04)
    cbar.set_label("Contribution")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image_np)
    plt.imshow(heatmap, cmap="bwr", vmin=-1, vmax=1, alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()