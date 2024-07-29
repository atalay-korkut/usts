import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

# Define the U-Net model (same as the one used for training)
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

# Load the model with saved weights
def load_model(model_path, device):
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Transformations for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Function to perform inference on a single image
def predict_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
    
    return output, original_size

# Function to extract bounding boxes from the predicted mask
def get_bounding_boxes(prediction, original_size, threshold=0.1, min_size=40, min_confidence=0.1):
    binary_mask = prediction > threshold
    labeled_mask, num_features = label(binary_mask)
    
    bboxes = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_mask == i)
        if coords.size == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
            continue
        
        bbox = (x_min, y_min, x_max, y_max)
        confidence = np.mean(prediction[y_min:y_max+1, x_min:x_max+1])
        
        if confidence < min_confidence:
            continue
        
        bboxes.append((bbox, confidence))
    
    # Sort the bounding boxes by confidence score in descending order
    bboxes.sort(key=lambda x: x[1], reverse=True)
    
    # Scale bounding boxes to the original image size
    bboxes_scaled = []
    for bbox, confidence in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min = int(x_min * original_size[0] / 256)
        y_min = int(y_min * original_size[1] / 256)
        x_max = int(x_max * original_size[0] / 256)
        y_max = int(y_max * original_size[1] / 256)
        bboxes_scaled.append(((x_min, y_min, x_max, y_max), confidence))
    
    return bboxes_scaled

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image_path, bboxes):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for bbox, confidence in bboxes:
        draw.rectangle(bbox, outline="red", width=2)
    return image

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/aorta-scan/Atalay/SAMUS/best_samus_unet_model.pth"  # Path to the saved model weights
    image_path = "/home/aorta-scan/Atalay/case1/images/image0.jpg"     # Path to the new image
    output_image_path = "bounding_boxes_image2.png"  # Path to save the image with bounding boxes

    model = load_model(model_path, device)
    prediction, original_size = predict_image(model, image_path, device)
    bboxes = get_bounding_boxes(prediction, original_size)
    
    for i, (bbox, confidence) in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        print(f"Bounding Box {i+1} Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, Confidence={confidence:.4f}")
    
    result_image = draw_bounding_boxes(image_path, bboxes)
    result_image.save(output_image_path)
    print(f"Image with bounding boxes saved to {output_image_path}")