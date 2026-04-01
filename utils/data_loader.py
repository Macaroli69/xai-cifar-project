import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import random

def load_data():
    #Convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    #Loads the CIFAR10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    #Loads the CIFAR10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    #Set random seed for reproducibility
    random.seed(43)

    #Choose random subset of data (Currently 50k Train, 10k Test)
    train_indices = random.sample(range(len(trainset)), 50000)
    test_indices = random.sample(range(len(testset)), 10000)

    #Make smaller datssets using the indices
    small_train = Subset(trainset, train_indices)
    small_test = Subset(testset, test_indices)

    #Feed data using DataLoader into the model in batches
    trainloader = DataLoader(small_train, batch_size=32, shuffle=True)
    testloader = DataLoader(small_test, batch_size=32, shuffle=False)

    return trainloader, testloader