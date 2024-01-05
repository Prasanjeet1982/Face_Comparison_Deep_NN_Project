import torch
import cv2
import numpy as np

# Assuming you have defined your SiameseNetwork model
from model import SiameseNetwork

# Load the trained model
model = SiameseNetwork()
model.load_state_dict(torch.load('siamese_model.pth'))
model.eval()

def preprocess_image(image_path):
    # Read and preprocess the input image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    img = np.transpose(img, (2, 0, 1))  # Transpose image dimensions
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return torch.tensor(img, dtype=torch.float32)

def compare_faces(image_path1, image_path2):
    # Preprocess the input images
    input1 = preprocess_image(image_path1)
    input2 = preprocess_image(image_path2)

    # Perform inference with the model
    with torch.no_grad():
        output1, output2 = model(input1, input2)
        similarity = torch.cosine_similarity(output1, output2).item()

    # Define a threshold for similarity
    threshold = 0.5  # Adjust as per your requirement

    # Compare the similarity against the threshold
    if similarity > threshold:
        return f"The faces in {image_path1} and {image_path2} are similar (Similarity: {similarity:.4f})"
    else:
        return f"The faces in {image_path1} and {image_path2} are different (Similarity: {similarity:.4f})"

if __name__ == "__main__":
    # Paths to the input images for comparison
    image_path1 = 'path_to_image1.jpg'
    image_path2 = 'path_to_image2.jpg'

    # Perform face comparison
    result = compare_faces(image_path1, image_path2)
    print(result)
