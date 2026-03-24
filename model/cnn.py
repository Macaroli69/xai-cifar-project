import torch.nn as nn
import torch.nn.functional as F

    #WHAT MODEL IS DOING:
    #Image → find edges → find shapes → shrink → flatten → classify
    #First two conv layers find edges and shapes, then pooling layers shrink the image, then fully connected layers flatten and classify the image

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        #First convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        #Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        #Pooling Layer (shrinks image by half)
        self.pool = nn.MaxPool2d(2, 2)

        #Fully connected layers (classification part)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        #Pass through first conv layer → activation → pooling
        x = self.pool(F.relu(self.conv1(x)))

        #Pass through second conv layer → activation → pooling
        x = self.pool(F.relu(self.conv2(x)))

        #Flatten image into a vector then through fully connected layer
        x = x.view(-1, 32 * 8 * 8)

        x = F.relu(self.fc1(x))

        #Final output (10 classes)
        x = self.fc2(x)

        return x
    
