import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lpips
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.cbam_denseunet import cbam_denseunet
# === Dataset Class ===
class cvccolondbsplitDataset(Dataset):
    def __init__(self, enhanced_dir, high_dir, transform=None):
        self.enhanced_dir = enhanced_dir
        self.high_dir = high_dir
        self.transform = transform
        self.image_names = os.listdir(enhanced_dir)
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
       enhanced_path = os.path.join(self.enhanced_dir, self.image_names[idx])
       high_path = os.path.join(self.high_dir, self.image_names[idx])
       enhanced_img = Image.open(enhanced_path).convert("RGB")
       high_img = Image.open(high_path).convert("RGB")
       if self.transform:
            enhanced_img = self.transform(enhanced_img)
            high_img = self.transform(high_img)
       return enhanced_img, high_img
# === SSIM Loss ===
def ssim_loss(img1, img2):
    total_ssim = 0.0
    batch_size = img1.size(0)
    for i in range(batch_size):
        img1_np = img1[i].cpu().detach().numpy().transpose(1, 2, 0)
        img2_np = img2[i].cpu().detach().numpy().transpose(1, 2, 0)
        img1_gray = cv2.cvtColor((img1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor((img2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        ssim_score = compare_ssim(img1_gray, img2_gray)
        total_ssim += 1 - ssim_score
    return total_ssim / batch_size
# === Edge Loss (Sobel) ===
def edge_loss(pred, target):
    def sobel(x):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        gray = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    return F.l1_loss(sobel(pred), sobel(target))
 # Total Loss Function
mse_loss_fn = nn.MSELoss()
lpips_loss_fn = lpips.LPIPS(net='vgg')  # initialized later on device
def total_loss_fn(pred, target, w_mse, w_ssim, w_lpips, w_edge, lpips_model, device):
        mse = mse_loss_fn(pred, target)
        ssim = ssim_loss(pred, target)
        lp = lpips_model(pred, target).mean()
        edge = edge_loss(pred, target)
        total = w_mse * mse + w_ssim * ssim + w_lpips * lp + w_edge * edge
        return total, mse, ssim, lp, edge
# === Folder Paths ===
train_enhanced_dir = "/content/outputs/train_enhanced"
train_high_dir = "/content/cvccolondbsplit/train/high"
val_enhanced_dir = "/content/outputs/val_enhanced"
val_high_dir = "/content/cvccolondbsplit/val/high"
# === Hyperparameters ===
learning_rate = 1e-4
weight_decay = 5e-5
num_epochs = 100
batch_size = 8
# === Loss Weights ===
w_mse = 0.4
w_ssim = 0.6
w_lpips = 0.8
w_edge = 0.2
# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_loss_fn = lpips_loss_fn.to(device)
# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# === Datasets and Loaders ===
train_dataset = cvccolondbDataset(train_enhanced_dir, train_high_dir, transform)
val_dataset = cvccolondbDataset(val_enhanced_dir, val_high_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Model ===
# Replace with your own model
model = cbam_denseunet().to(device)
# === Optimizer and Scheduler ===
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# === Early Stopping ===
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
# === Loss Tracking ===
train_losses = []
val_losses = []
# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for input_img, target_img in train_loader:
        input_img, target_img = input_img.to(device), target_img.to(device)
        optimizer.zero_grad()
        output = model(input_img)
        total_loss, mse, ssim_val, lpips_val, edge_val = total_loss_fn(output, target_img,w_mse, w_ssim, w_lpips, w_edge, lpips_loss_fn, device)
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
        avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_img, target_img in val_loader:
            input_img, target_img = input_img.to(device), target_img.to(device)
            output = model(input_img)
            total_loss, _, _, _, _ = total_loss_fn(output, target_img,w_mse, w_ssim, w_lpips, w_edge, lpips_loss_fn, device)
            val_loss += total_loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")
# === Scheduler Step ===
    scheduler.step(avg_val_loss)
        # === Early Stopping Check ===
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # torch.save(model.state_dict(), "best_model.pth")  # Optional save
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
# === Plot the loss curves ===
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
