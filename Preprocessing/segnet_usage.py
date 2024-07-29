import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

class SegNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False):
        super(SegNet, self).__init__()
        vgg_bn = models.vgg16_bn(pretrained=pretrained)
        encoder = list(vgg_bn.features.children())

        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:], nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1))
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder, self.stage4_decoder, self.stage5_decoder)
        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

# Load the model with saved weights
def load_model(model_path, device):
    model = SegNet(num_classes=1)
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
def get_bounding_boxes(prediction, original_size, threshold=0.2, min_size=4, min_confidence=0.2):
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
    model_path = "/home/aorta-scan/Atalay/SAMUS/best_segnet_model.pth"  # Path to the saved model weights
    image_path = "/home/aorta-scan/Atalay/case1/images/image0.jpg"  # Path to the new image
    output_image_path = "segnet_bounding_boxes_image.png"  # Path to save the image with bounding boxes

    model = load_model(model_path, device)
    prediction, original_size = predict_image(model, image_path, device)
    bboxes = get_bounding_boxes(prediction, original_size)
    
    for i, (bbox, confidence) in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        print(f"Bounding Box {i+1} Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, Confidence={confidence:.4f}")
    
    result_image = draw_bounding_boxes(image_path, bboxes)
    result_image.save(output_image_path)
    print(f"Image with bounding boxes saved to {output_image_path}")