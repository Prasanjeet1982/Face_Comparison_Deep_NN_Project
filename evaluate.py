import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Assuming you have defined your SiameseNetwork model and dataset class
from model import SiameseNetwork
from custom_dataset import CustomDataset

# Initialize Siamese Network model
model = SiameseNetwork()

# Load the trained model
model.load_state_dict(torch.load('siamese_model.pth'))
model.eval()

# Assuming you have defined your dataset and dataloader for testing
test_dataset = CustomDataset(train=False)  # Assuming you have a separate test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the evaluation criterion (e.g., Contrastive Loss, Triplet Loss, etc.)
criterion = nn.CosineEmbeddingLoss()  # You may need to choose an appropriate loss for your task

# Evaluation loop
total_loss = 0.0
correct = 0
total_samples = 0

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data  # Assuming your dataset returns inputs and labels
        
        # Forward pass
        outputs1, outputs2 = model(inputs[0], inputs[1])  # Siamese network forward pass
        loss = criterion(outputs1, outputs2, labels)  # Calculate the loss
        
        # Compute accuracy
        similarity = torch.cosine_similarity(outputs1, outputs2)
        predicted = similarity > 0.5  # Threshold for similarity, you can adjust this
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Aggregate loss
        total_loss += loss.item()

# Calculate average loss and accuracy
average_loss = total_loss / len(test_loader)
accuracy = correct / total_samples

print(f"Average Loss: {average_loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")
