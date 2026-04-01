# WHAT INTEGRATED GRADIENTS IS DOING:
# 1. Take one test image
# 2. Compare it to a baseline image (all zeros)
# 3. Measure how each pixel contributes to the prediction
# 4. Build an attribution map from those contributions
# 5. Show which parts of the image matter most

import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients


# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_integrated_gradients_map(model, image_tensor):
    model.eval()

    # Add batch dimension
    input_tensor = image_tensor.unsqueeze(0)

    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    # Create Integrated Gradients object
    ig = IntegratedGradients(model)

    # Use a black image as baseline
    baseline = torch.zeros_like(input_tensor)

    # Compute attributions
    attributions, _ = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=predicted_label,
        return_convergence_delta=True
    )

    # Convert 3-channel attribution map into a 2D heatmap
    heatmap = attributions.squeeze().detach().numpy()
    heatmap = np.mean(np.abs(heatmap), axis=0)

    # Normalize heatmap
    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    return heatmap, predicted_label


def explain_with_integrated_gradients(model, image_tensor, true_label):
    heatmap, predicted_label = get_integrated_gradients_map(model, image_tensor)

    image_np = image_tensor.permute(1, 2, 0).detach().numpy()

    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    # Heatmap only
    plt.subplot(1, 3, 2)
    ig_plot = plt.imshow(heatmap, cmap="jet")
    plt.title(f"Integrated Gradients\nPred: {class_names[predicted_label]}")
    plt.axis("off")

    cbar = plt.colorbar(ig_plot, fraction=0.046, pad=0.04)
    cbar.set_label("Importance")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image_np)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()