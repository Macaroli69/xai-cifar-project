import os
import torch
import random
import numpy as np

from model.cnn import SimpleCNN
from utils.data_loader import load_data, get_random_images
from utils.train import train_model
from utils.visualization import show_combined_explanations

MODEL_PATH = "saved_models/simple_cnn.pth"

# Set to an integer like 42 for reproducibility
# Set to None for different random images each run
SEED = None


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def main():
    # =========================
    # 1. Optional reproducibility
    # =========================
    set_seed(SEED)

    # =========================
    # 2. Load dataset
    # =========================
    trainloader, testloader = load_data()
    testset = testloader.dataset

    # =========================
    # 3. Create model
    # =========================
    model = SimpleCNN()

    # Make sure saved_models folder exists
    os.makedirs("saved_models", exist_ok=True)

    # =========================
    # 4. Train or load model
    # =========================
    choice = input("Type 'T' to train a new model or 'L' to use saved model: (T/L) ").strip().lower()

    if choice == "t":
        print("Training new model...")
        model = train_model(model, trainloader)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    elif choice == "l":
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print(f"Loaded model from {MODEL_PATH}")

    else:
        print("Invalid choice. Please type 'T' or 'L'.")
        return

    # =========================
    # 5. Pick random test images
    # =========================
    num_images = 3
    images, labels = get_random_images(testset, num_images)

    # =========================
    # 6. Show explanations
    # =========================
    for i in range(num_images):
        print(f"\n--- Image {i + 1} ---")

        image_tensor = images[i]
        true_label = labels[i]

        print("Showing combined explanations...")
        show_combined_explanations(model, image_tensor, true_label)


if __name__ == "__main__":
    main()