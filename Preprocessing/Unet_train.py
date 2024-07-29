import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
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

# Dataset class with support for multiple directories
class SegmentationDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None):
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform
        
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            image_files = [f for f in os.listdir(image_dir) if self.is_image_file(f)]
            mask_files = [f for f in os.listdir(mask_dir) if self.is_image_file(f)]
            common_files = set(image_files).intersection(set(mask_files))
            
            for common_file in common_files:
                self.image_paths.append(os.path.join(image_dir, common_file))
                self.mask_paths.append(os.path.join(mask_dir, common_file))
                
                # Print warning if there are image files without corresponding mask files
                if common_file not in mask_files:
                    print(f"Warning: Mask file not found for image {common_file}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

    def is_image_file(self, filename):
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return any(filename.lower().endswith(ext) for ext in extensions)

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the dataset from multiple directories
image_dirs = ["/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TN3K/img", "/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TG3K/img","/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/DDTI/img","/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/BUSI/img"] 
mask_dirs = ["/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TN3K/label", "/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TG3K/label","/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TG3K/label","/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/BUSI/label"]

dataset = SegmentationDataset(image_dirs, mask_dirs, transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25
best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

    # Save the model if it has the best validation loss so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_samus_unet_model.pth")
        print(f"Model saved at epoch {epoch+1}")

# Save the final model
torch.save(model.state_dict(), "final_samus_unet_model.pth")
print("Final model saved.")
