# xai-cifar-project

This project explores how explainable AI (XAI) methods help visualize what a CNN is focusing on when making image classification predictions on the CIFAR-10 dataset.

The goal is not only to generate explanations, but also to evaluate how **consistent** those explanations are across repeated runs.

---

## Model

The model used is a simple Convolutional Neural Network (CNN) built in PyTorch.

It classifies images into 10 classes:

- airplane  
- automobile  
- bird  
- cat  
- deer  
- dog  
- frog  
- horse  
- ship  
- truck  

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

- `main.py` → Runs the model and displays explanations for random test images  
- `consistency_test.py` → Runs consistency evaluation across multiple images and runs  

### Model
- `model/cnn.py` → CNN architecture  

### Utilities
- `utils/data_loader.py` → Loads CIFAR-10 and handles random sampling  
- `utils/train.py` → Model training  
- `utils/consistency_utils.py` → Consistency metrics  
- `utils/visualization.py` → Combined explanation visualization  

### Methods
- `methods/gradcam.py` → Grad-CAM  
- `methods/lime_explain.py` → LIME  
- `methods/shap_explain.py` → SHAP (signed heatmaps)  
- `methods/integrated_gradients.py` → Integrated Gradients  

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## How to Run

### 1. Train or Load Model

```bash
python main.py
```

You will be prompted:

* `T` → Train a new model
* `L` → Load an existing model

⚠️ If this is your first run, you **must train a model**

The model will be saved to:

```bash
saved_models/simple_cnn.pth
```

---

### 2. Run Consistency Testing

```bash
python consistency_test.py
```

This will evaluate:

* 30 images
* 10 runs per method
* All four XAI methods

---

## Consistency Evaluation

The project evaluates consistency using **two complementary metrics**.

---

### 1. Cosine Similarity (Global Consistency)

Each explanation map is flattened into a vector.

For each pair of runs:

* Compute cosine similarity
* Average all pairwise scores

**What it measures:**

* Overall similarity of the heatmap
* Whether the general explanation pattern stays the same

**Limitation:**

* Does not focus on *where* important regions are
* Small spatial shifts may still appear similar

---

### 2. Top-K IoU Overlap (Region Consistency)

For each heatmap:

* Select top **10% most important pixels**
* Convert into a binary mask

For each pair of runs:

* Compute Intersection over Union (IoU)

**What it measures:**

* Whether the **same important regions** are selected
* Spatial stability of explanations

**Why this matters:**

Two heatmaps can look similar overall but highlight different regions.
IoU captures this difference.

---

## Why These Metrics Together Are Strong

Using both metrics gives a more complete view:

* Cosine similarity → global similarity
* IoU overlap → spatial consistency

This allows us to distinguish between:

* Methods that look similar overall
* Methods that consistently highlight the same regions

---

## Observed Behavior

Typical results show:

* **Grad-CAM** → highly consistent (deterministic)
* **Integrated Gradients** → highly consistent
* **SHAP** → highly consistent in this implementation
* **LIME** → less consistent, especially in top-k regions

### LIME Behavior

LIME often shows:

* High cosine similarity (similar overall shape)
* Lower IoU (different exact regions)

This suggests:

* Globally stable explanations
* Locally unstable important regions

---

## Key Takeaways

* Consistency does **not** mean correctness
* A method can be perfectly consistent but still produce poor explanations
* Stability is only one aspect of explanation quality

This project focuses on **repeatability and stability**, not correctness.

---

## Notes

* CIFAR-10 dataset is used
* The CNN is intentionally simple for fast experimentation
* Grad-CAM and Integrated Gradients are deterministic
* LIME introduces randomness
* SHAP behaves deterministically in this setup

---

## Project Status

This project is nearing completion.

Completed:

* Model + all XAI methods implemented
* Consistency metrics (cosine + IoU) implemented
* Project structure cleaned

Remaining:

* Final analysis and write-up
* Eventually runng all these test on tiny-Imagenet data set (higher resolution images 64x64)
```
