import argparse
import json
import os
import torch
from torchvision import transforms
from PIL import Image

# === Argument Parser ===
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--phase", type=str, required=True, help="Phase: train / test / val")
parser.add_argument("-c", "--config", type=str, required=True, help="Path to JSON config file")
args = parser.parse_args()

# === Load Config ===
with open(args.config, "r") as f:
    config = json.load(f)

input_dir = config["input_dir"]
output_dir = config["output_dir"]
model_path = config["model_path"]
image_size = tuple(config.get("image_size", [256, 256]))
num_epochs = config.get("epochs", 5)
batch_size = config.get("batch_size", 8)
lr = config.get("learning_rate", 1e-4)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load CBAM-RDB-UNet Model ===
from models.cbam_rdb_unet import CBAM_RDB_UNet
model = CBAM_RDB_UNet(in_channels=3, out_channels=1).to(device)

# === Load Model Weights If Available ===
if os.path.exists(model_path) and args.phase in ["test", "val"]:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model weights from: {model_path}")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

# === Simple Dataset Loader ===
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader

class ImageFolderDataset(Dataset):
    def __init__(self, input_folder, transform=None):
        self.image_paths = [os.path.join(input_folder, fname)
                            for fname in os.listdir(input_folder)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.loader(self.image_paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.image_paths[idx])

# === Train Function ===
def train_model():
    os.makedirs(output_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    dataset = ImageFolderDataset(input_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            targets = imgs.clone()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"📘 Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")

# === Test/Validation Function ===
def evaluate_images(tag="test"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for fname in os.listdir(input_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(input_dir, fname)
                img = Image.open(img_path).convert('RGB')
                inp = transform(img).unsqueeze(0).to(device)
                out = model(inp).squeeze().cpu().clamp(0, 1)
                out_img = to_pil(out)
                out_img.save(os.path.join(output_dir, fname))
                print(f"✅ {tag.capitalize()} Enhanced: {fname}")
    print(f"🎉 {tag.capitalize()} enhancement complete. Saved to: {output_dir}")

# === Run Phase ===
if args.phase == "train":
    train_model()
elif args.phase == "test":
    evaluate_images("test")
elif args.phase == "val":
    evaluate_images("val")
else:
    raise NotImplementedError(f"Phase '{args.phase}' is not supported.")
