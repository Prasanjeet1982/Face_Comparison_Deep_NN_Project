import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images, target_size=(224, 224)):
    processed_images = []
    for img in images:
        # Resize the image to the target size
        resized_img = cv2.resize(img, target_size)
        
        # Normalize pixel values between 0 and 1
        normalized_img = resized_img / 255.0
        
        # Optional: Convert to grayscale if needed
        # gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)
        
        processed_images.append(normalized_img)
    return processed_images

def save_processed_images(processed_images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for i, img in enumerate(processed_images):
        cv2.imwrite(os.path.join(output_folder, f"processed_image_{i}.jpg"), img * 255.0)

if __name__ == "__main__":
    # Replace 'path_to_input_folder' with the path to your input images folder
    input_folder = 'path_to_input_folder'
    # Replace 'path_to_output_folder' with the desired output location for processed images
    output_folder = 'path_to_output_folder'
    
    # Load images from the input folder
    images = load_images_from_folder(input_folder)
    
    # Preprocess the loaded images
    processed_images = preprocess_images(images)
    
    # Save the preprocessed images to the output folder
    save_processed_images(processed_images, output_folder)
