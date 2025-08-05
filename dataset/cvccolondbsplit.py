import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class cvccolondbsplit(Dataset):
    def __init__(self, low_dir, high_dir, transform=True, paired=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform_enabled = transform
        self.paired = paired
        self.low_images = sorted(os.listdir(low_dir))
        self.high_images = sorted(os.listdir(high_dir))
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return min(len(self.low_images), len(self.high_images))
    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])
        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")
        if self.transform_enabled:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
        else:
            low_img = transforms.ToTensor()(low_img)
            high_img = transforms.ToTensor()(high_img)
        return {
            'low': low_img,
            'high': high_img,
            'low_path': low_path,
            'high_path': high_path
        }
