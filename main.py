from model.cnn import SimpleCNN
from utils.data_loader import load_data
from utils.train import train_model

def main():
    #Step 1: Load dataset
    trainloader, testloader = load_data()

    #Step 2: Create our CNN model
    model = SimpleCNN()

    #Step 3: Train the model
    print("Training model...")
    model = train_model(model, trainloader)

    print("Done training!")

#Double check that this is the main module being run
if __name__ == "__main__":
    main()
    