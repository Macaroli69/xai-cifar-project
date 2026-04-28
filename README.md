# xai-cifar-project

This project explores how explainable AI (XAI) methods help visualize what a CNN is focusing on when making image classification predictions on the CIFAR-10 dataset.

The goal is not only to generate explanations, but also to evaluate how **consistent** those explanations are across repeated runs.

Consistency does not mean correctness.
A method can be perfectly consistent but still produce poor explanations.

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

## Dataset Choice (Why CIFAR-10)

The CIFAR-10 dataset was intentionally chosen because it contains **low-resolution images (32x32 pixels)**.

This project focuses on evaluating the **consistency of explanation methods under constrained visual conditions**.

Low-resolution images:

- reduce fine detail and texture information  
- make object boundaries less clear  
- increase ambiguity in model attention  

This makes consistency testing more meaningful because:

- small changes in explanations become more noticeable  
- methods must remain stable even when visual information is limited  
- differences between XAI methods are easier to observe  

In other words, CIFAR-10 provides a controlled setting to evaluate how reliable explanation methods are when working with **minimal visual information**.

---

### Methods

This project uses four XAI methods to visualize what parts of an image the model is focusing on when making predictions.

Each method produces a heatmap that highlights important regions of the image.

- Grad-CAM  
  Uses gradients flowing into the final convolutional layer to highlight important spatial regions.  
  The visualization shows which parts of the image most influenced the model’s decision.

- LIME  
  Perturbs parts of the image and observes how predictions change.  
  The visualization highlights regions that most affect the prediction when altered.
  It also provides a rough segmentation of the image, showing the boundaries the model considers meaningful.

- Integrated Gradients  
  Computes feature importance by accumulating gradients from a baseline image to the input.  
  The visualization shows how each pixel contributes to the final prediction.

- SHAP  
  Uses game theory-based attribution to assign importance values to pixels.  
  It evaluates how the prediction changes when parts of the image are removed, indicating whether those regions help or hurt the model’s decision.

SHAP explanations use **signed values**:

- Positive values → support the predicted class  
- Negative values → go against the predicted class  

Additionally, SHAP is evaluated under **slightly varying approximation settings** (such as masking and evaluation parameters).  

This introduces controlled variability across runs, allowing SHAP to be compared more fairly with sampling-based methods like LIME in consistency experiments.  

The core SHAP method is unchanged, but this setup reflects how explanation stability behaves under different approximation conditions.

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
- `utils/consistency_runner.py` → Runs repeated XAI evaluations and computes metrics  
- `utils/consistency_display.py` → Handles formatted terminal output  
- `utils/results_saver.py` → Saves CSV outputs into structured folders  

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
```
---

## How to Run

### 1. Train or Load Model
```bash
python main.py
```
You will be prompted:

- T → Train a new model 
- L → Load an existing model

⚠️ If this is your first run, you must train a model  

The model will be saved to:

`saved_models/simple_cnn.pth`

---

### 2. Run Consistency Testing
```bash
python consistency_test.py
```
This will run consistency evaluation using:

- 100 images  
- 7 runs per method  
- All four XAI methods  

---

### Terminal Output

For each image, the program prints:

- True label  
- Predicted label  
- Whether the prediction is correct  
- Cosine similarity scores  
- Top-k IoU scores  

---

### 3. Saving Results

After the test finishes, you will be prompted:

`Save results to CSV? (y/n):`

If you choose yes, two files will be created inside a structured results folder:

`results/100_images_7_runs/consistency_summary_MM-DD-YYYY_HH-MM-SS.csv`

- Contains final average consistency scores for each method  

`results/100_images_7_runs/consistency_detailed_MM-DD-YYYY_HH-MM-SS.csv`

- Contains per-image results including:
  - true label  
  - predicted label  
  - correctness  
  - cosine similarity scores  
  - top-k IoU scores  

This folder structure organizes results by experiment settings and prevents overwriting previous runs.

---

## Configuration Options

You can adjust key parts of the project to control behavior and runtime.

---

`main.py`

- SEED = None  
  - Set to a number (e.g., 42) for reproducible image selection  
  - Set to None to allow random image selection each run  

- num_images = 3  
  - Number of image examples displayed  
  - Each example shows a full set of visualizations for all four methods (Grad-CAM, LIME, Integrated Gradients, SHAP)    

---

`consistency_test.py`

- NUM_IMAGES = 100 → Number of test images  
- NUM_RUNS = 7 → Repeats per method (higher = slower but more stable)  
- TOP_PERCENT = 0.10 → Percentage of important pixels used for IoU  

---

`methods/lime_explain.py`

- num_samples = 1000 → LIME sampling size (higher = more stable, slower)

---

`methods/shap_explain.py`

- max_evals = 100 → SHAP evaluation budget (higher = more precise, slower)

---

These settings let you balance:

- speed vs accuracy  
- reproducibility vs randomness  
- coarse vs detailed explanations  
- runtime vs stability  

---

## Consistency Evaluation

The project evaluates consistency using two complementary metrics.

---

### 1. Cosine Similarity (Global Consistency)

Each explanation map is flattened into a vector.

For each pair of runs:

- Compute cosine similarity  
- Average all pairwise scores  

What it measures:

- Overall similarity of the heatmap  
- Whether the general explanation pattern stays the same  

Limitation:

- Does not focus on where important regions are  
- Small spatial shifts may still appear similar  

---

### 2. Top-K IoU Overlap (Region Consistency)

For each heatmap:

- Select top 10 percent most important pixels  
- Convert into a binary mask  

For each pair of runs:

- Compute Intersection over Union (IoU)  

What it measures:

- Whether the same important regions are selected  
- Spatial stability of explanations  

Why this matters:

Two heatmaps can look similar overall but highlight different regions.  
IoU captures this difference.

---

## Why These Metrics Together Are Strong

Using both metrics gives a more complete view:

- Cosine similarity → global similarity  
- IoU overlap → spatial consistency  

---

## Observed Behavior

Typical results show (From a 1k Image 15 Runs TEST):

- Grad-CAM → highly consistent (deterministic)  
- Integrated Gradients → highly consistent  
- SHAP → moderately consistent  
- LIME → least consistent, especially in top-k regions  

When separating results by model prediction outcome:

- Grad-CAM and Integrated Gradients remain stable across both correct and incorrect predictions  
- LIME shows slightly lower consistency on incorrect predictions  
- SHAP shows slight variation, with some cases exhibiting higher consistency on incorrect predictions  

Overall, differences between methods are more significant than differences between correct and incorrect predictions.

---

## Notes

- CIFAR-10 dataset is used intentionally for low-resolution testing  
- The CNN is intentionally simple for fast experimentation  
- Grad-CAM and Integrated Gradients are deterministic  
- LIME introduces randomness  
- SHAP uses approximation-based variability in this setup  
- Consistency is evaluated separately for all images, correct predictions, and incorrect predictions  
- Explanation behavior can vary slightly between correct and incorrect predictions, depending on the method.

---

## Project Status

This project is complete.
