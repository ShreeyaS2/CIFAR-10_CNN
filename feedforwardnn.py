import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def generate_model():
    model = nn.Sequential(
        # First Block
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Second Block
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Classifier
        nn.Flatten(),
        nn.Linear(in_features=64 * 7 * 7, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=10) 
    )
    return model

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Standard normalization
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

model = generate_model().to(device)
criterion = nn.CrossEntropyLoss() # Standard for classification
optimizer = optim.Adam(model.parameters(), lr=0.001) # The Adam optimizer you asked about!

#training function
def train_network(epochs=500):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get data (Note the Python 3 next() logic we discussed earlier happens inside the loop)
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass (Backpropagation)
            loss.backward()

            # Optimize (Update weights)
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499: # Print every 500 batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}: loss {running_loss / 500:.3f}')
                running_loss = 0.0

    print('Finished Training')

def visualize_prediction(model, testloader, device):
    # Set model to evaluation mode (turns off dropout/batchnorm)
    model.eval()
    
    # Get a single batch of images from the test_loader
    # We use the next(iter()) pattern we discussed earlier
    images, labels = next(iter(testloader))

    # Move data to the same device as the model (CPU or GPU)
    images, labels = images.to(device), labels.to(device)
    
    # Make a prediction
    with torch.no_grad(): # Disable gradient calculation for speed/memory
        outputs = model(images)
        # Get the index of the highest value (the class prediction)
        _, predicted = torch.max(outputs, 1)

    # Pick the first image in the batch to display
    img = images[0].cpu() # Move back to CPU for plotting
    label = labels[0].cpu().item()
    prediction = predicted[0].cpu().item()

    # Unnormalize the image for display
    img = img / 2 + 0.5     
    npimg = img.numpy()
    
    # Reshape the image 
    # PyTorch is (Channels, Height, Width), but Matplotlib needs (Height, Width, Channels)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Show title with color-coding
    color = "green" if prediction == label else "red"
    plt.title(f"Actual: {classes[label]}\nPredicted: {classes[prediction]}", color=color)
    plt.axis('off') # Hide the X/Y axes
    plt.show()
    
# Execute
if __name__ == "__main__":
    train_network()
    visualize_prediction(model, testloader, device)