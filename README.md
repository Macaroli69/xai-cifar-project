# xai-cifar-project

This project explores how explainable AI (XAI) methods help visualize what a CNN is focusing on when making image classification predictions on the CIFAR-10 dataset.

The goal is not only to generate explanations, but also to evaluate how **consistent and stable** those explanations are across repeated runs.

---

## Model
The model used is a simple Convolutional Neural Network (CNN) built in PyTorch.

It classifies images into 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## XAI Methods Compared
The project compares four explanation methods:

- Grad-CAM
- LIME
- Integrated Gradients
- SHAP

### SHAP Update
SHAP explanations use **signed values**:
- Positive values → support the predicted class
- Negative values → go against the predicted class

This provides more meaningful explanations than using only absolute importance.

---

## Project Files
- `main.py`  
  Runs the model and displays explanations for random test images.

- `consistency_test.py`  
  Runs consistency evaluation across multiple images and multiple runs.

- `model/cnn.py`  
  Defines the CNN model.

- `utils/data_loader.py`  
  Loads CIFAR-10 and provides random sampling utilities.

- `utils/train.py`  
  Trains the model.

- `utils/consistency_utils.py`  
  Contains similarity and overlap metrics used for consistency evaluation.

- `utils/visualization.py`  
  Displays combined explanation outputs for all methods.

- `methods/gradcam.py`  
  Grad-CAM implementation.

- `methods/lime_explain.py`  
  LIME implementation.

- `methods/shap_explain.py`  
  SHAP implementation (signed heatmaps).

- `methods/integrated_gradients.py`  
  Integrated Gradients implementation.

---

## Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

## How to run
1. Train or load the model

Run:
```bash
python main.py
```
You will be asked:

Type T to train a new model
Type L to load the saved model

If you train a model, it will be saved to:

```bash
saved_models/simple_cnn.pth
```

2. Run consistency testing

After a model has been trained and saved, run:
```bash
python consistency_test.py
```
This will test:

30 images
10 runs per method

Grad-CAM consistency
LIME consistency
SHAP consistency
Integrated Gradients

## Consistency Evaluation
The project evaluates consistency using two complementary metrics.

1. Cosine Similarity (Global Consistency)

Each explanation map is flattened into a vector.

For each pair of runs:

cosine similarity is computed
all pairwise scores are averaged
What it measures:
Overall similarity of the full heatmap
Whether the general explanation pattern stays the same
Limitation:
Does not focus on where the most important regions are
Small spatial shifts may still appear very similar

2. Top-K IoU Overlap (Region Consistency)

For each heatmap:

top 10% most important pixels are selected
converted into a binary mask

For each pair of runs:

Intersection over Union (IoU) is computed
What it measures:
Whether the same important regions are selected across runs
Spatial stability of explanations
Why this is important:

Two heatmaps can look similar overall but highlight different exact regions.
IoU captures this difference.

## Why These Metrics Together Are Strong
Using both metrics gives a more complete view of consistency:

Cosine similarity → captures global similarity
IoU overlap → captures spatial stability

This combination allows us to distinguish between:

Methods that look similar overall
Methods that consistently highlight the same regions

## Observed Behavior
Typical results show:

Grad-CAM → highly consistent (deterministic)
Integrated Gradients → highly consistent
SHAP → highly consistent in this implementation
LIME → less consistent, especially in top-k regions

LIME often shows:

high cosine similarity (similar overall shape)
lower IoU (different exact important regions)

This suggests LIME explanations are:

globally stable
but locally less consistent

## Key Takeaways
Consistency does not mean correctness
A method can be perfectly consistent but still produce poor explanations
Stability is only one aspect of explanation quality

This project focuses specifically on repeatability and stability of explanations, not their accuracy.

## Notes
CIFAR-10 is used as the dataset
The CNN is intentionally simple for fast experimentation
Grad-CAM and Integrated Gradients are deterministic in this setup
LIME introduces randomness due to sampling
SHAP behaves deterministically under current settings

