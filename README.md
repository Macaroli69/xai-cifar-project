# xai-cifar-project

This project looks at how explainable AI methods can help show what a CNN is focusing on when it makes image classification predictions on CIFAR 10.

The model used is a simple CNN built in PyTorch. The explainability methods compared in this project are:

- Grad CAM
- LIME
- SHAP

The project also includes a consistency evaluation script that runs each explanation method multiple times on the same images and compares how similar the explanation maps are.

## Project Files

- `main.py`  
  Train the CNN or load a saved model, then show Grad CAM, LIME, and SHAP explanations on test images.

- `consistency_test.py`  
  Run consistency testing for Grad CAM, LIME, and SHAP over multiple images and multiple runs.

- `model/cnn.py`  
  Defines the simple CNN used for CIFAR 10 classification.

- `utils/data_loader.py`  
  Loads the CIFAR 10 dataset.

- `utils/train.py`  
  Trains the CNN model.

- `utils/consistency_utils.py`  
  Contains cosine similarity and average pairwise similarity functions for explanation comparison.

- `methods/gradcam.py`  
  Generates Grad CAM heatmaps.

- `methods/lime_explain.py`  
  Generates LIME explanations.

- `methods/shap_explain.py`  
  Generates SHAP explanations.

## Requirements

Install dependencies with:

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

saved_models/simple_cnn.pth

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

It also separates results into:

all images
correct predictions only
incorrect predictions only

## Consistency Method

Consistency is measured by comparing explanation maps from multiple runs of the same method on the same image.

For each image, the explanation method is run multiple times. Then every pair of explanation maps is compared using cosine similarity.

All pairwise similarity scores are averaged to get a final consistency score for that image.

A higher score means the method gives more similar explanations across runs, while a lower score means the explanations change more between runs.

## Notes
The dataset used is CIFAR 10
The model is a simple CNN for classifying 10 image classes
Grad CAM is deterministic, so it will usually have very high consistency
LIME and SHAP can vary more depending on random sampling
