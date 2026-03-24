import torch
import torch.nn as nn
import torch.optim as optim

#WHAT TRAINING IS DOING:
#For each batch
#Batch → predict → compare with real answer → calculate error → adjust weights
#Repeat and model improves

#Can change epochs and learning rate here
def train_model(model, trainloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0

        #Reset gradients, forward pass, calculate loss, backward pass, and update weights for each batch
        for images, labels in trainloader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            #Loss tracking
            running_loss += loss.item()

        #Print average loss for the epoch
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.3f}")

    return model