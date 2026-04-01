import os
import torch

from model.cnn import SimpleCNN
from utils.data_loader import load_data
from utils.train import train_model
from methods.gradcam import show_gradcam
from methods.lime_explain import explain_with_lime
from methods.shap_explain import explain_with_shap

MODEL_PATH = "saved_models/simple_cnn.pth"


def main():
    # Step 1: Load dataset
    trainloader, testloader = load_data()

    # Step 2: Create CNN model
    model = SimpleCNN()

    # Make sure saved_models folder exists
    os.makedirs("saved_models", exist_ok=True)

    # Step 3: Ask user whether to train a new model or load saved model
    choice = input("Type 'T' to train a new model or 'L' to use saved model: (T/L) ").strip().lower()

    if choice == "t":
        print("Training new model...")
        model = train_model(model, trainloader)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    elif choice == "l":
        if not os.path.exists(MODEL_PATH):
            print(f"Saved model not found at {MODEL_PATH}")
            print("Train the model first by choosing 'T'.")
            return

        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()
        print(f"Loaded model from {MODEL_PATH}")

    else:
        print("Invalid choice. Please type 'T' or 'L'.")
        return

    # Step 4: Get one batch of test images
    images, labels = next(iter(testloader))

    # Show the next 3 images
    num_images = 3

    for i in range(num_images):
        print(f"\n--- Image {i + 1} ---")

        image_tensor = images[i]
        true_label = labels[i].item()

        print("Showing Grad-CAM visualization...")
        show_gradcam(model, image_tensor, true_label)

        print("Showing LIME explanation...")
        explain_with_lime(model, image_tensor, true_label)

        print("Showing SHAP explanation...")
        explain_with_shap(model, image_tensor, true_label)


if __name__ == "__main__":
    main()