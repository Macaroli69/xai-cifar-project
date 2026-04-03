import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_data(batch_size=32):
    # Convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Feed data using DataLoader into the model in batches
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def get_random_images(dataset, num_images):
    indices = random.sample(range(len(dataset)), num_images)

    images = []
    labels = []

    for idx in indices:
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)

    return images, labels