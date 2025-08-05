import torch
import argparse
import os
import json
from torch.utils.data import DataLoader
from models import cbam_rdb_unet
from dataset import cvccolondbsplit
from loss_utils import TotalLoss
from torchvision.utils import save_image

# --------- Argument Parser ---------
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], required=True, help='Mode: train | val | test')
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
args = parser.parse_args()

# --------- Load Config ---------
with open(args.config, 'r') as f:
    config = json.load(f)

# --------- Initialize Model ---------
model_args = config["model"]["which_model"]["args"]
model = cbam_rdb_unet(**model_args).cuda()

# --------- Train ---------
if args.mode == 'train':
    dataset_args = config["train"]["dataset"]["args"]
    dataloader_args = config["train"]["dataloader"]["args"]

    train_data = cvccolondbsplit(**dataset_args)
    train_loader = DataLoader(train_data, **dataloader_args)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    loss_fn = TotalLoss()
    n_epoch = config["train"]["n_epoch"]

    for epoch in range(n_epoch):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            low, high = batch["low"].cuda(), batch["high"].cuda()
            optimizer.zero_grad()
            output = model(low)
            loss = loss_fn(output, high)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[Epoch {epoch+1}/{n_epoch}] Loss: {running_loss / len(train_loader):.4f}")

    os.makedirs(config["train"]["model_path"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config["train"]["model_path"], config["train"]["model_name"]))
    print("✅ Training complete and model saved.")

# --------- Validation ---------
elif args.mode == 'val':
    model.load_state_dict(torch.load(os.path.join(config["train"]["model_path"], config["train"]["model_name"])))
    model.eval()

    dataset_args = config["val"]["dataset"]["args"]
    dataloader_args = config["val"]["dataloader"]["args"]

    val_data = cvccolondbsplit(**dataset_args)
    val_loader = DataLoader(val_data, **dataloader_args)

    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            low, high = batch["low"].cuda(), batch["high"].cuda()
            output = model(low)
            total_loss += torch.nn.functional.mse_loss(output, high).item()

    print(f"✅ Validation MSE: {total_loss / len(val_loader):.4f}")

# --------- Test ---------
elif args.mode == 'test':
    model.load_state_dict(torch.load(os.path.join(config["test"]["model_path"], config["test"]["model_name"])))
    model.eval()

    dataset_args = config["test"]["dataset"]["args"]
    dataloader_args = config["test"]["dataloader"]["args"]

    test_data = cvccolondbsplit(**dataset_args)
    test_loader = DataLoader(test_data, **dataloader_args)

    os.makedirs(config["test"]["output_images_path"], exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            low = batch["low"].cuda()
            output = model(low)
            save_image(output, os.path.join(config["test"]["output_images_path"], f"output_{i}.png"))

    print(f"✅ Test images saved to: {config['test']['output_images_path']}")


