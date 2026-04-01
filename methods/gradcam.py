# WHAT GRAD-CAM IS DOING:
# 1. Take one test image
# 2. Run it through the trained CNN
# 3. Look at the last conv layer (conv2)
# 4. Find what parts of the image mattered most
# 5. Show a heatmap

import torch
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import LayerGradCam
from captum.attr import LayerAttribution


# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def show_gradcam(model, image_tensor, true_label):
    model.eval()

    # Add batch dimension
    input_image = image_tensor.unsqueeze(0)

    # Get model prediction
    output = model(input_image)
    predicted_label = torch.argmax(output, dim=1).item()

    # Grad-CAM uses the last convolution layer
    gradcam = LayerGradCam(model, model.conv2)

    # Create attribution map for the predicted class
    attribution = gradcam.attribute(input_image, target=predicted_label)

    # Resize attribution map so it matches the image size
    upsampled_attr = LayerAttribution.interpolate(
        attribution,
        input_image.shape[2:]
    )

    # Convert to numpy for plotting
    image_np = input_image.squeeze().permute(1, 2, 0).detach().numpy()
    heatmap = upsampled_attr.squeeze().detach().numpy()

    # Keep positive values only
    heatmap = np.maximum(heatmap, 0)

    # Normalize heatmap
    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    # Heatmap only
    plt.subplot(1, 3, 2)
    heatmap_plot = plt.imshow(heatmap, cmap="jet")
    plt.title(f"Grad-CAM\nPred: {class_names[predicted_label]}")
    plt.axis("off")

    # Colorbar
    cbar = plt.colorbar(heatmap_plot, fraction=0.046, pad=0.04)
    cbar.set_label("Importance")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image_np)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()