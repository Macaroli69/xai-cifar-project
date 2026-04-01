# WHAT LIME IS DOING:
# 1. Take one test image
# 2. Break image into regions (superpixels)
# 3. Hide different parts of the image many times
# 4. See how the model prediction changes
# 5. Highlight the regions that matter most

import numpy as np
import torch
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
from functools import partial


# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Better segmentation for tiny CIFAR-10 images
segmenter = partial(slic, n_segments=20, compactness=1, sigma=1)


def get_lime_map(model, image_tensor):
    model.eval()

    # Add batch dimension for prediction
    input_tensor = image_tensor.unsqueeze(0)

    # Convert tensor image to numpy image for LIME
    image_np = image_tensor.permute(1, 2, 0).numpy()

    # LIME prediction function
    def predict_fn(images_batch):
        model.eval()

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

    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation
    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=segmenter
    )

    # Get highlighted image and mask
    temp, mask = explanation.get_image_and_mask(
        label=predicted_label,
        positive_only=False,
        num_features=6,
        hide_rest=False
    )

    # Convert mask to float map so it can be compared later
    lime_map = mask.astype(np.float32)

    # Normalize to 0 to 1 if possible
    if lime_map.max() != lime_map.min():
        lime_map = (lime_map - lime_map.min()) / (lime_map.max() - lime_map.min())

    return lime_map, temp, mask, predicted_label


def explain_with_lime(model, image_tensor, true_label):
    lime_map, temp, mask, predicted_label = get_lime_map(model, image_tensor)

    image_np = image_tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(f"Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    # LIME regions
    plt.subplot(1, 3, 2)
    plt.imshow(temp)
    plt.title(f"LIME Regions\nPred: {class_names[predicted_label]}")
    plt.axis("off")

    # Boundaries on original image
    plt.subplot(1, 3, 3)
    plt.imshow(mark_boundaries(image_np, mask))
    plt.title("LIME Boundaries")
    plt.axis("off")

    plt.tight_layout()
    plt.show()