import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import numpy as np

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

# Data augmentation and transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
])

# Load the dataset from multiple directories
image_dirs = ["/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TN3K/img", "/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TG3K/img","/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/DDTI/img","/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/BUSI/img"] 
mask_dirs = ["/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TN3K/label", "/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TG3K/label","/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/ThyroidNodule-TG3K/label","/home/aorta-scan/Atalay/SAMUS/dataset/SAMUS/BUSI/label"]


dataset = SegmentationDataset(image_dirs, mask_dirs, transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Adjust batch size and num_workers for faster training
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegNet(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25
best_val_loss = float('inf')
patience = 5
trigger_times = 0

# Training loop with early stopping
for epoch in range(num_epochs):
    start_time = time.time()
    
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
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Time: {epoch_time:.2f}s")

    # Save the model if it has the best validation loss so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_segnet_model.pth")
        print(f"Model saved at epoch {epoch+1}")
        trigger_times = 0  # Reset the early stopping counter
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# Save the final model
torch.save(model.state_dict(), "final_segnet_model.pth")
print("Final model saved.")