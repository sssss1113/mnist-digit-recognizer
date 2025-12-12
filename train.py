import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Net

BATCH_SIZE = 64
EPOCHS = 3
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

train_loader = DataLoader(train_data, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True
        )

net = Net().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

for epoch in range(EPOCHS):
    net.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS} loss: {total_loss:.3f}")
    
os.makedirs("models", exist_ok=True)    
torch.save(net.state_dict(), "models/mnist_cnn.pth")
print("Model saved as models/mnist_cnn.pth")