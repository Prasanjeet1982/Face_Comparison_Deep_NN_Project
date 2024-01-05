import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming you have defined your SiameseNetwork model and dataset class
from model import SiameseNetwork
from custom_dataset import CustomDataset

# Hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Initialize Siamese Network model
model = SiameseNetwork()

# Assuming you have defined your dataset and dataloader
train_dataset = CustomDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data  # Assuming your dataset returns inputs and labels
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs1, outputs2 = model(inputs[0], inputs[1])  # Siamese network forward pass
        loss = criterion(outputs1, outputs2, labels)  # Calculate the loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Batch [{i + 1}/{len(train_loader)}], "
                  f"Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

print("Training finished")
# Save the trained model
torch.save(model.state_dict(), 'siamese_model.pth')
